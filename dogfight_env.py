# dogfight_env.py
import os
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dogfight_core import (
    WIDTH, HEIGHT, GROUND_Y, CEILING_Y, GRAVITY, BULLET_SPEED, BULLET_DAMAGE, HIT_RADIUS,
    Jet, FastNumpyPolicy, modular_dx, wrap_angle, extract_rl_obs,
    easy_naive_ai_control, boom_zoom_bot_control, kiter_bot_control, hard_naive_ai_control, floor_hugger_bot_control
)

class DogfightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_type="hard", max_steps=1500):
        super().__init__()
        self.opponent_type = opponent_type
        self.max_steps = max_steps

        self.action_space = spaces.MultiDiscrete([3, 2, 2])
        self.observation_space = spaces.Box(low=-2.5, high=2.5, shape=(28,), dtype=np.float32)

        self.p1 = None
        self.p2 = None
        self.bullets = []
        self.step_count = 0
        self.active_opp_mode = "hard"
        self.pool_models = []
        self.active_opp_model = None
        

    def set_stage_config(self, opp_type, pool_paths):
        self.opponent_type = opp_type
        self.pool_models = [FastNumpyPolicy(p) for p in pool_paths if os.path.exists(p)]

    def _get_obs(self):
        obs = np.array(extract_rl_obs(self.p1, self.p2, self.bullets), dtype=np.float32)
        return np.clip(obs, -2.5, 2.5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.bullets = []

        eval_fixed = options.get("fixed", False) if options else False
        if eval_fixed:
            x1, y1, a1 = 220.0, 220.0, 0.0
            x2, y2, a2 = 980.0, 500.0, math.pi
        else:
            x1 = random.uniform(100, WIDTH - 100)
            y1 = random.uniform(GROUND_Y + 150, CEILING_Y - 120)
            a1 = random.uniform(-math.pi, math.pi)

            offset_x = random.choice([-1, 1]) * random.uniform(260, 560)
            x2 = (x1 + offset_x) % WIDTH
            y2 = random.uniform(GROUND_Y + 150, CEILING_Y - 120)
            a2 = random.uniform(-math.pi, math.pi)

        self.p1 = Jet(x1, y1, a1, (52, 152, 219), "RL AGENT")
        self.p2 = Jet(x2, y2, a2, (231, 76, 60), "OPPONENT")

        if not eval_fixed:
            spd1 = random.uniform(5.0, 11.0)
            spd2 = random.uniform(5.0, 11.0)
            self.p1.vx, self.p1.vy = spd1 * math.cos(a1), spd1 * math.sin(a1)
            self.p2.vx, self.p2.vy = spd2 * math.cos(a2), spd2 * math.sin(a2)

        if self.opponent_type == "mixed":
            r = random.random()
            if r < 0.15:
                self.active_opp_mode = "easy"
            elif r < 0.45:
                self.active_opp_mode = "kiter"
            elif r < 0.70:
                self.active_opp_mode = "bnz"
            else:
                self.active_opp_mode = "hard"
        elif self.opponent_type == "self_play" and len(self.pool_models) > 0:
             r = random.random()
             if r < 0.20:
                 self.active_opp_mode = "floor"   # Teaches safe low-pass strafing
             elif r < 0.40:
                 self.active_opp_mode = "kiter"
             elif r < 0.55:
                 self.active_opp_mode = "hard"
             elif r < 0.70:
                 self.active_opp_mode = "bnz"
             else:
                 self.active_opp_mode = "model"
                 self.active_opp_model = random.choice(self.pool_models)
        else:
            self.active_opp_mode = self.opponent_type

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        p1_turn = int(action[0]) - 1
        p1_thr = int(action[1])
        p1_fire = int(action[2])

        if self.active_opp_mode == "easy":
            p2_turn, p2_thr, p2_fire = easy_naive_ai_control(self.p2, self.p1)
        elif self.active_opp_mode == "bnz":
            p2_turn, p2_thr, p2_fire = boom_zoom_bot_control(self.p2, self.p1)
        elif self.active_opp_mode == "kiter":
            p2_turn, p2_thr, p2_fire = kiter_bot_control(self.p2, self.p1)
        elif self.active_opp_mode == "model" and self.active_opp_model is not None:
            opp_obs = np.clip(np.array(extract_rl_obs(self.p2, self.p1, self.bullets), dtype=np.float32), -2.5, 2.5)
            opp_act = self.active_opp_model.predict(opp_obs, deterministic=False)
            p2_turn, p2_thr, p2_fire = int(opp_act[0]) - 1, int(opp_act[1]), int(opp_act[2])
        elif self.active_opp_mode == "floor":
            p2_turn, p2_thr, p2_fire = floor_hugger_bot_control(self.p2, self.p1)
        else:
            p2_turn, p2_thr, p2_fire = hard_naive_ai_control(self.p2, self.p1)

        prev_hp1 = self.p1.hp
        prev_hp2 = self.p2.hp

        fired_now = self.p1.update(p1_turn, p1_thr, p1_fire, self.bullets)
        self.p2.update(p2_turn, p2_thr, p2_fire, self.bullets)

        surviving = []
        for b in self.bullets:
            if not b.update():
                continue
            hit = False
            for target in [self.p1, self.p2]:
                if b.owner is not target and target.alive:
                    dx = modular_dx(b.x, target.x)
                    dy = target.y - b.y
                    if math.hypot(dx, dy) < HIT_RADIUS:
                        target.hp = max(0, target.hp - BULLET_DAMAGE)
                        hit = True
                        if target.hp == 0:
                            target.alive = False
                        break
            if not hit:
                surviving.append(b)
        self.bullets = surviving

        reward = -0.002

        dmg_dealt = prev_hp2 - self.p2.hp
        dmg_taken = prev_hp1 - self.p1.hp
        reward += dmg_dealt * 0.45
        reward -= dmg_taken * 0.35

        dx = modular_dx(self.p1.x, self.p2.x)
        dy = self.p2.y - self.p1.y
        dist = math.hypot(dx, dy)
        t_hit = min(38.0, dist / BULLET_SPEED)
        lead_dx = modular_dx(self.p1.x, self.p2.x + self.p2.vx * t_hit - self.p1.vx * t_hit * 0.35)
        lead_dy = (self.p2.y + self.p2.vy * t_hit - self.p1.vy * t_hit * 0.35) - self.p1.y
        lead_err = abs(wrap_angle(math.atan2(lead_dy, lead_dx) - self.p1.angle))

        # Allow firing out to 820 px so RL can counter long-range kiters!
        if dist < 820.0:
            reward += 0.020 * max(0.0, math.cos(lead_err))
            if lead_err < 0.14:
                reward += 0.025

        if fired_now and (lead_err > 0.45 or dist > 820.0):
            reward -= 0.020
        if self.p1.overheated:
            reward -= 0.04

        pred_y_30 = self.p1.y + self.p1.vy * 30.0 - 0.5 * GRAVITY * 900.0
        ground_danger = max(0.0, min(1.0, (GROUND_Y + 145.0 - pred_y_30) / 145.0))
        if ground_danger > 0.0:
             reward -= (0.28 * (ground_danger ** 2)) + (0.12 * ground_danger)
        if self.p1.y < GROUND_Y + 70.0 and self.p1.vy < 0:
             reward -= 0.35

        if self.p1.y > CEILING_Y - 25.0:
            reward -= 0.05

        opp_aim_err = abs(wrap_angle(math.atan2(-dy, -dx) - self.p2.angle))
        if dist < 550.0 and opp_aim_err < 0.35:
            reward -= 0.025

        terminated = not self.p1.alive or not self.p2.alive
        truncated = self.step_count >= self.max_steps

        win = 0
        opp_crashed = self.p2.crashed
        if terminated:
            if self.p1.alive and not self.p2.alive:
                reward += 12.0 if opp_crashed else 30.0
                win = 1
            elif self.p2.alive and not self.p1.alive:
                reward -= 25.0
                if self.p1.crashed:
                    reward -= 18.0
                win = -1
        elif truncated:
            reward += (self.p1.hp - self.p2.hp) * 0.12
            if self.p1.hp > self.p2.hp:
                win = 1
            elif self.p2.hp > self.p1.hp:
                win = -1

        info = {
            "win": win, "p1_hp": self.p1.hp, "p2_hp": self.p2.hp,
            "opp_crashed": opp_crashed, "self_crashed": self.p1.crashed
        }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info
