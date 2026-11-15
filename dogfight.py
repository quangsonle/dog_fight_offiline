# dogfight_offline.py
import sys
import os
import math
import random
import numpy as np
import pygame
from stable_baselines3 import PPO

from dogfight_core import (
    WIDTH, HEIGHT, GROUND_Y, CEILING_Y, MAX_HEAT, BULLET_DAMAGE, HIT_RADIUS, V_STALL,
    Jet, Bullet, wrap_angle, modular_dx, rotate_point,
    extract_rl_obs, hard_naive_ai_control
)

# =====================================================================
# VISUAL FX & TARGET
# =====================================================================
class FixedTarget:
    def __init__(self, avoid_x, avoid_y):
        self.radius = 22
        self.x, self.y = 0, 0
        self.respawn(avoid_x, avoid_y)

    def respawn(self, avoid_x, avoid_y):
        for _ in range(50):
            tx = random.uniform(100, WIDTH - 100)
            ty = random.uniform(GROUND_Y + 110, CEILING_Y - 80)
            if math.hypot(modular_dx(avoid_x, tx), ty - avoid_y) > 260:
                self.x, self.y = tx, ty
                return
        self.x, self.y = (avoid_x + WIDTH / 2) % WIDTH, HEIGHT * 0.55

    def draw(self, screen):
        sx, sy = int(self.x), int(HEIGHT - self.y)
        pygame.draw.circle(screen, (230, 70, 70), (sx, sy), self.radius)
        pygame.draw.circle(screen, (245, 245, 250), (sx, sy), int(self.radius * 0.68))
        pygame.draw.circle(screen, (230, 70, 70), (sx, sy), int(self.radius * 0.36))
        pygame.draw.circle(screen, (255, 215, 70), (sx, sy), 4)

class Particle:
    def __init__(self, x, y, vx, vy, color, life, radius):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.color, self.life, self.max_life, self.radius = color, life, life, radius

    def update(self):
        self.x = (self.x + self.vx) % WIDTH
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, screen):
        alpha = int(255 * (self.life / self.max_life))
        surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
        screen.blit(surf, (int(self.x) - self.radius, int(HEIGHT - self.y) - self.radius))

def spawn_explosion(x, y, particles, color, count=25):
    for _ in range(count):
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(1.0, 6.5)
        particles.append(Particle(x, y, spd * math.cos(ang), spd * math.sin(ang), color, random.randint(15, 35), random.randint(2, 5)))

def draw_jet(screen, jet, show_guides=False):
    if not jet.alive:
        return
    sx, sy = int(jet.x), int(HEIGHT - jet.y)

    if show_guides:
        for d in range(35, 320, 30):
            px = (jet.x + d * math.cos(jet.angle)) % WIDTH
            py = jet.y + d * math.sin(jet.angle)
            if GROUND_Y < py < HEIGHT:
                pygame.draw.circle(screen, (90, 140, 180), (int(px), int(HEIGHT - py)), 2)

        spd = jet.speed()
        if spd > 1.0:
            fpm_x = int((jet.x + (jet.vx / spd) * 55) % WIDTH)
            fpm_y = int(HEIGHT - (jet.y + (jet.vy / spd) * 55))
            if abs(fpm_x - sx) < 100:
                pygame.draw.circle(screen, (100, 230, 140), (fpm_x, fpm_y), 5, 1)

    local_pts = [(18, 0), (-12, 10), (-7, 0), (-12, -10)]
    screen_pts = []
    for lx, ly in local_pts:
        rx, ry = rotate_point(lx, ly, jet.angle)
        screen_pts.append((sx + rx, sy - ry))

    pygame.draw.polygon(screen, jet.color, screen_pts)
    pygame.draw.polygon(screen, (240, 245, 255), screen_pts, 2)
    cx, cy = rotate_point(5, 0, jet.angle)
    pygame.draw.circle(screen, (200, 240, 255), (int(sx + cx), int(sy - cy)), 3)

def draw_bullet(screen, b):
    sx, sy = int(b.x), int(HEIGHT - b.y)
    ex, ey = int((b.x - b.vx * 0.7) % WIDTH), int(HEIGHT - (b.y - b.vy * 0.7))
    if abs(sx - ex) < 100:
        pygame.draw.line(screen, (255, 240, 140), (sx, sy), (ex, ey), 3)
    pygame.draw.circle(screen, b.color, (sx, sy), 2)

def draw_hud(screen, jet, x_pos, font, small_font, align_right=False, is_training=False, targets_hit=0, shots_fired=0):
    bar_w = 240
    bx = x_pos - bar_w if align_right else x_pos

    if not is_training:
        pygame.draw.rect(screen, (40, 45, 55), (bx, 20, bar_w, 18), border_radius=4)
        hp_w = int(bar_w * max(0, jet.hp) / 100.0)
        hp_col = jet.color if jet.hp > 30 else (230, 60, 60)
        pygame.draw.rect(screen, hp_col, (bx, 20, hp_w, 18), border_radius=4)
        pygame.draw.rect(screen, (200, 210, 220), (bx, 20, bar_w, 18), 1, border_radius=4)
        label = font.render(f"{jet.name}  HP: {jet.hp}", True, (240, 245, 255))
        lbl_x = (x_pos - label.get_width()) if align_right else bx
        screen.blit(label, (lbl_x, 42))
    else:
        label = font.render(f"FLIGHT SCHOOL  |  Targets Hit: {targets_hit}  |  Shots: {shots_fired}", True, (110, 230, 160))
        screen.blit(label, (bx, 20))

    spd = jet.speed()
    stall_warn = " [STALL!]" if spd < V_STALL else ""
    s_col = (255, 80, 80) if spd < V_STALL else (170, 190, 210)
    spd_surf = small_font.render(f"Airspeed: {spd * 45:.0f} kts  |  Alt: {int(jet.y - GROUND_Y)}m{stall_warn}", True, s_col)
    spd_x = (x_pos - spd_surf.get_width()) if align_right else bx
    screen.blit(spd_surf, (spd_x, 66 if not is_training else 48))

    heat_bar_x = (x_pos - 140) if align_right else bx
    pygame.draw.rect(screen, (40, 45, 55), (heat_bar_x, 88 if not is_training else 70, 140, 8), border_radius=2)
    h_w = int(140 * (jet.heat / MAX_HEAT))
    h_col = (255, 60, 60) if jet.overheated else (255, 190, 60)
    pygame.draw.rect(screen, h_col, (heat_bar_x, 88 if not is_training else 70, h_w, 8), border_radius=2)
    heat_lbl = small_font.render("OVERHEAT!" if jet.overheated else "GUN HEAT", True, h_col)
    hl_x = (heat_bar_x - heat_lbl.get_width() - 8) if align_right else (heat_bar_x + 148)
    screen.blit(heat_lbl, (hl_x, 84 if not is_training else 66))

# =====================================================================
# RL MODEL REGISTRY
# =====================================================================
def load_available_models():
    candidates = [
    ("RL Champion (Self-Play)", "models/rl_champion.zip"),
    ("RL Stage 2 (vs Bots)", "models/rl_vs_bots.zip"),
    ("RL Stage 1 (Starter)", "models/rl_stage1.zip")
]
    loaded = []
    for label, path in candidates:
        if os.path.exists(path):
            loaded.append((label, PPO.load(path)))
    return loaded

def rl_predict(model, jet, opp, bullets):
    obs = np.clip(np.array(extract_rl_obs(jet, opp, bullets), dtype=np.float32), -2.5, 2.5)
    act, _ = model.predict(obs, deterministic=True)
    return int(act[0]) - 1, int(act[1]), int(act[2])

# =====================================================================
# MAIN GAME LOOP
# =====================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Kinetic Dogfight: Human vs. Rule-Bot vs. Self-Play RL")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Arial", 14)
    big_font = pygame.font.SysFont("Arial", 24, bold=True)
    title_font = pygame.font.SysFont("Arial", 38, bold=True)

    available_models = load_available_models()
    selected_model_idx = 0

    app_state = "MENU"   # MENU, PLAY, GAMEOVER
    game_mode = 1        # 1: Train, 2: 1P vs Hard, 3: 1P vs RL, 4: RL vs Hard, 5: RL vs RL, 6: 2P Local
    mouse_fly_by_wire = True
    sim_speed = 1        # 1x or 3x for spectator watching
    waiting_to_launch = True

    p1, p2 = None, None
    target_obj = None
    targets_hit, shots_fired = 0, 0
    bullets, particles = [], []
    winner_text = ""
    spectator_score = {"p1": 0, "p2": 0, "draws": 0}
    round_steps = 0

    def start_match(mode, reset_tally=False):
        nonlocal app_state, game_mode, waiting_to_launch, p1, p2, target_obj, targets_hit, shots_fired, bullets, particles, winner_text, round_steps
        if reset_tally:
            spectator_score["p1"], spectator_score["p2"], spectator_score["draws"] = 0, 0, 0
        game_mode = mode
        round_steps = 0
        # Spectator AI modes launch immediately; Human modes wait for input
        waiting_to_launch = (mode in [1, 2, 3, 6])

        p1_name = "PLAYER 1 (BLUE)"
        if mode in [4, 5]:
            m_name = available_models[selected_model_idx][0] if available_models else "NO RL MODEL"
            p1_name = f"BLUE: {m_name}"

        p1 = Jet(380, 220, math.pi * 0.5, (52, 152, 219), p1_name)
        bullets, particles = [], []
        winner_text = ""

        if mode == 1:
            p2 = None
            target_obj = FixedTarget(p1.x, p1.y)
            targets_hit, shots_fired = 0, 0
        else:
            target_obj = None
            if mode == 2 or mode == 4:
                p2_name = "HARD RULE-BASED AI"
            elif mode == 3:
                m_name = available_models[selected_model_idx][0] if available_models else "NO RL MODEL"
                p2_name = m_name
            elif mode == 5:
                p2_name = "RED: SELF-PLAY CLONE"
            else:
                p2_name = "PLAYER 2 (RED)"
            p2 = Jet(820, 220, math.pi * 0.5, (231, 76, 60), p2_name)

        app_state = "PLAY"

    while True:
        # ==================== MENU ====================
        if app_state == "MENU":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB and available_models:
                        selected_model_idx = (selected_model_idx + 1) % len(available_models)
                    elif event.key == pygame.K_c:
                        mouse_fly_by_wire = not mouse_fly_by_wire
                    elif event.key == pygame.K_1:
                        start_match(1, True)
                    elif event.key == pygame.K_2:
                        start_match(2, True)
                    elif event.key == pygame.K_3 and available_models:
                        start_match(3, True)
                    elif event.key == pygame.K_4 and available_models:
                        start_match(4, True)
                    elif event.key == pygame.K_5 and available_models:
                        start_match(5, True)
                    elif event.key == pygame.K_6:
                        start_match(6, True)

            screen.fill((18, 22, 32))
            screen.blit(title_font.render("2D KINETIC DOGFIGHT ARENA", True, (240, 245, 255)), (70, 32))

            model_status = f"Active RL Model [TAB]: {available_models[selected_model_idx][0]}" if available_models else "No RL Models Found (Run: python train_dogfight.py)"
            m_col = (100, 230, 150) if available_models else (255, 110, 110)
            screen.blit(font.render(model_status, True, m_col), (72, 82))

            ctrl_str = "1P Controls [C to toggle]: MOUSE FLY-BY-WIRE (Point to steer, Left-Click=Fire, Right-Click/W=Thrust)" if mouse_fly_by_wire else "1P Controls [C to toggle]: KEYBOARD ([A/D or Left/Right]=Turn, [W/Up]=Thrust, [Space]=Fire)"
            screen.blit(small_font.render(ctrl_str, True, (255, 210, 90)), (72, 110))

            menu_items = [
                (1, "[1] FLIGHT SCHOOL: Target Practice (Never Ends)", (46, 204, 113), True, "Practice flying & aiming with safe bounce floor and sightline."),
                (2, "[2] 1P vs. HARD RULE-BASED AI", (52, 152, 219), True, "Challenge the deterministic lead-pursuit & break-turn bot."),
                (3, "[3] 1P vs. TRAINED RL AGENT", (155, 89, 182), bool(available_models), "Fight the trained PPO neural network policy yourself."),
                (4, "[4] AI SHOWDOWN: RL Agent vs. Hard Rule-Based AI", (230, 126, 34), bool(available_models), "Spectate the trained RL model battling the Hard Rule-Based AI!"),
                (5, "[5] SELF-PLAY: RL Agent vs. RL Agent", (241, 196, 15), bool(available_models), "Spectate two trained neural networks dogfighting each other."),
                (6, "[6] 2-PLAYER LOCAL DUEL (Same Keyboard)", (231, 76, 60), True, "P1: [A/D/W + Space]   |   P2: [Arrows + Enter]")
            ]

            for i, (m_id, m_title, m_c, enabled, m_desc) in enumerate(menu_items):
                cy = 148 + i * 82
                bg_col = (28, 34, 48) if enabled else (22, 25, 34)
                txt_col = (240, 245, 255) if enabled else (90, 95, 110)
                pygame.draw.rect(screen, bg_col, (70, cy, WIDTH - 140, 70), border_radius=8)
                pygame.draw.rect(screen, m_c if enabled else (60, 65, 75), (70, cy, 8, 70), border_top_left_radius=8, border_bottom_left_radius=8)
                screen.blit(big_font.render(m_title, True, txt_col), (98, cy + 10))
                screen.blit(small_font.render(m_desc, True, (160, 175, 195) if enabled else (80, 85, 95)), (98, cy + 42))

            screen.blit(small_font.render("Press [1-6] to launch | [TAB] Cycle RL Model | [C] Toggle Mouse/Keyboard Steering", True, (130, 145, 165)), (72, 655))
            pygame.display.flip()
            clock.tick(60)
            continue

        # ==================== PLAY / GAMEOVER ====================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    app_state = "MENU"
                elif event.key == pygame.K_r:
                    start_match(game_mode, False)
                elif event.key == pygame.K_c:
                    mouse_fly_by_wire = not mouse_fly_by_wire
                elif event.key == pygame.K_f and game_mode in [4, 5]:
                    sim_speed = 3 if sim_speed == 1 else 1
                elif event.key == pygame.K_TAB and available_models:
                    selected_model_idx = (selected_model_idx + 1) % len(available_models)
                    start_match(game_mode, False)

        keys = pygame.key.get_pressed()
        mouse_btns = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()

        if app_state == "PLAY" and waiting_to_launch:
            flight_keys = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_SPACE, pygame.K_f]
            if any(keys[k] for k in flight_keys) or any(mouse_btns):
                waiting_to_launch = False

        steps_this_frame = sim_speed if (app_state == "PLAY" and game_mode in [4, 5]) else 1

        for _ in range(steps_this_frame):
            if app_state == "PLAY" and not waiting_to_launch:
                round_steps += 1
                fx_events = []

                # Player 1 / Left Agent Controls
                if game_mode in [1, 2, 3]:
                    if mouse_fly_by_wire:
                        dx_m = mx - p1.x
                        dy_m = (HEIGHT - my) - p1.y
                        target_ang = math.atan2(dy_m, dx_m)
                        err_m = wrap_angle(target_ang - p1.angle)
                        p1_turn = 1 if err_m > 0.06 else (-1 if err_m < -0.06 else 0)
                        p1_thr = 1 if (mouse_btns[2] or keys[pygame.K_w] or keys[pygame.K_UP] or keys[pygame.K_LSHIFT]) else 0
                        p1_fire = 1 if (mouse_btns[0] or keys[pygame.K_SPACE] or keys[pygame.K_f]) else 0
                    else:
                        p1_turn = (1 if (keys[pygame.K_LEFT] or keys[pygame.K_a]) else 0) + (-1 if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) else 0)
                        p1_thr = 1 if (keys[pygame.K_UP] or keys[pygame.K_w]) else 0
                        p1_fire = 1 if (keys[pygame.K_SPACE] or keys[pygame.K_f]) else 0
                elif game_mode in [4, 5]:
                    p1_turn, p1_thr, p1_fire = rl_predict(available_models[selected_model_idx][1], p1, p2, bullets)
                else:  # 2P Local
                    p1_turn = (1 if keys[pygame.K_a] else 0) + (-1 if keys[pygame.K_d] else 0)
                    p1_thr = 1 if keys[pygame.K_w] else 0
                    p1_fire = 1 if (keys[pygame.K_f] or keys[pygame.K_SPACE]) else 0

                fired_now = p1.update(p1_turn, p1_thr, p1_fire, bullets, safe_floor=(game_mode == 1), fx_events=fx_events)
                if game_mode == 1 and fired_now:
                    shots_fired += 1

                # Opponent Controls
                if game_mode in [2, 3, 4, 5, 6]:
                    if game_mode in [2, 4]:
                        p2_turn, p2_thr, p2_fire = hard_naive_ai_control(p2, p1)
                    elif game_mode in [3, 5]:
                        p2_turn, p2_thr, p2_fire = rl_predict(available_models[selected_model_idx][1], p2, p1, bullets)
                    else:
                        p2_turn = (1 if keys[pygame.K_LEFT] else 0) + (-1 if keys[pygame.K_RIGHT] else 0)
                        p2_thr = 1 if keys[pygame.K_UP] else 0
                        p2_fire = 1 if (keys[pygame.K_RETURN] or keys[pygame.K_RSHIFT]) else 0
                    p2.update(p2_turn, p2_thr, p2_fire, bullets, safe_floor=False, fx_events=fx_events)

                # Spawn Particles from FX events
                for ev_type, ex, ey, evx, evy in fx_events:
                    if ev_type == "flame":
                        particles.append(Particle(ex, ey, evx, evy, (255, 160, 40), 10, 3))
                    elif ev_type == "vapor":
                        particles.append(Particle(ex, ey, 0, 0, (180, 200, 220), 16, 2))
                    elif ev_type == "smoke":
                        particles.append(Particle(ex, ey, evx, evy, (110, 110, 120), 25, 4))
                    elif ev_type == "bounce":
                        spawn_explosion(ex, ey, particles, (100, 220, 255), 8)
                    elif ev_type == "crash":
                        spawn_explosion(ex, ey, particles, (255, 120, 40), 40)

                # Bullets & Hit Check
                surviving = []
                for b in bullets:
                    if not b.update():
                        continue
                    hit = False
                    if game_mode == 1 and target_obj is not None:
                        dx = modular_dx(b.x, target_obj.x)
                        dy = target_obj.y - b.y
                        if math.hypot(dx, dy) < target_obj.radius + 5.0:
                            targets_hit += 1
                            spawn_explosion(target_obj.x, target_obj.y, particles, (255, 200, 60), 30)
                            target_obj.respawn(p1.x, p1.y)
                            hit = True
                    else:
                        for target in [p1, p2]:
                            if b.owner is not target and target.alive:
                                dx = modular_dx(b.x, target.x)
                                dy = target.y - b.y
                                if math.hypot(dx, dy) < HIT_RADIUS:
                                    target.hp = max(0, target.hp - BULLET_DAMAGE)
                                    spawn_explosion(b.x, b.y, particles, (255, 220, 100), 10)
                                    hit = True
                                    if target.hp == 0:
                                        target.alive = False
                                        spawn_explosion(target.x, target.y, particles, target.color, 50)
                                    break
                    if not hit:
                        surviving.append(b)
                bullets = surviving

                # Match End Check
                if game_mode != 1 and (not p1.alive or not p2.alive or round_steps >= 2400):
                    app_state = "GAMEOVER"
                    if p1.alive and not p2.alive:
                        winner_text = f"{p1.name} WINS!"
                        spectator_score["p1"] += 1
                    elif p2.alive and not p1.alive:
                        winner_text = f"{p2.name} WINS!"
                        spectator_score["p2"] += 1
                    elif p1.hp > p2.hp:
                        winner_text = f"{p1.name} WINS ON POINTS!"
                        spectator_score["p1"] += 1
                    elif p2.hp > p1.hp:
                        winner_text = f"{p2.name} WINS ON POINTS!"
                        spectator_score["p2"] += 1
                    else:
                        winner_text = "DRAW!"
                        spectator_score["draws"] += 1
                    break

        particles = [pt for pt in particles if pt.update()]

        # ==================== RENDER ====================
        screen.fill((18, 22, 32))

        for alt in [150, 300, 450, 600]:
            gy = int(HEIGHT - alt)
            pygame.draw.line(screen, (28, 34, 46), (0, gy), (WIDTH, gy), 1)
            screen.blit(small_font.render(f"{alt - int(GROUND_Y)}m", True, (60, 70, 90)), (8, gy - 16))

        pygame.draw.line(screen, (45, 55, 75), (4, 0), (4, HEIGHT - int(GROUND_Y)), 2)
        pygame.draw.line(screen, (45, 55, 75), (WIDTH - 4, 0), (WIDTH - 4, HEIGHT - int(GROUND_Y)), 2)

        floor_col = (35, 55, 70) if game_mode == 1 else (35, 42, 38)
        line_col = (70, 170, 210) if game_mode == 1 else (80, 120, 90)
        pygame.draw.rect(screen, floor_col, (0, HEIGHT - int(GROUND_Y), WIDTH, int(GROUND_Y)))
        pygame.draw.line(screen, line_col, (0, HEIGHT - int(GROUND_Y)), (WIDTH, HEIGHT - int(GROUND_Y)), 3)

        for pt in particles:
            pt.draw(screen)
        for b in bullets:
            draw_bullet(screen, b)

        if target_obj is not None:
            target_obj.draw(screen)

        # Draw Mouse Cursor Reticle in Fly-By-Wire mode
        if game_mode in [1, 2, 3] and mouse_fly_by_wire:
            pygame.draw.circle(screen, (120, 200, 255), (mx, my), 10, 1)
            pygame.draw.line(screen, (120, 200, 255), (mx - 14, my), (mx + 14, my), 1)
            pygame.draw.line(screen, (120, 200, 255), (mx, my - 14), (mx, my + 14), 1)

        draw_jet(screen, p1, show_guides=(game_mode == 1))
        if p2 is not None:
            draw_jet(screen, p2, show_guides=False)

        if game_mode == 1:
            draw_hud(screen, p1, 20, font, small_font, is_training=True, targets_hit=targets_hit, shots_fired=shots_fired)
        else:
            draw_hud(screen, p1, 20, font, small_font, align_right=False)
            draw_hud(screen, p2, WIDTH - 20, font, small_font, align_right=True)

        # Spectator Tally Bar
        if game_mode in [4, 5]:
            tally_str = f"TALLY  Blue: {spectator_score['p1']}  |  Red: {spectator_score['p2']}  |  Draws: {spectator_score['draws']}   [Speed: {sim_speed}x]"
            tsurf = font.render(tally_str, True, (255, 220, 90))
            screen.blit(tsurf, (WIDTH // 2 - tsurf.get_width() // 2, 22))

        if game_mode in [1, 2, 3]:
            c_txt = "MOUSE STEERING [C to toggle]" if mouse_fly_by_wire else "KEYBOARD STEERING [C to toggle]"
            bottom_msg = f"{c_txt}  |  [R] Restart  |  [M] Menu"
        elif game_mode in [4, 5]:
            bottom_msg = "[F] Toggle 1x/3x Speed  |  [TAB] Switch RL Checkpoint  |  [R] Next Round  |  [M] Menu"
        else:
            bottom_msg = "[R] Rematch  |  [M] Menu"

        b_surf = small_font.render(bottom_msg, True, (110, 125, 145))
        screen.blit(b_surf, (WIDTH // 2 - b_surf.get_width() // 2, HEIGHT - 28))

        if app_state == "PLAY" and waiting_to_launch:
            msg = "MOVE MOUSE OR PRESS ANY FLIGHT KEY TO LAUNCH!" if (game_mode in [1, 2, 3] and mouse_fly_by_wire) else "PRESS ANY FLIGHT KEY TO LAUNCH!"
            m_surf = big_font.render(msg, True, (255, 220, 80))
            box_w = m_surf.get_width() + 50
            overlay = pygame.Surface((box_w, 70))
            overlay.set_alpha(210)
            overlay.fill((22, 26, 36))
            screen.blit(overlay, (WIDTH // 2 - box_w // 2, HEIGHT // 2 - 120))
            pygame.draw.rect(screen, (255, 220, 80), (WIDTH // 2 - box_w // 2, HEIGHT // 2 - 120, box_w, 70), 2)
            screen.blit(m_surf, (WIDTH // 2 - m_surf.get_width() // 2, HEIGHT // 2 - 100))

        if app_state == "GAMEOVER":
            overlay = pygame.Surface((680, 160))
            overlay.set_alpha(230)
            overlay.fill((22, 26, 36))
            screen.blit(overlay, (WIDTH // 2 - 340, HEIGHT // 2 - 80))
            pygame.draw.rect(screen, (200, 210, 225), (WIDTH // 2 - 340, HEIGHT // 2 - 80, 680, 160), 2)

            w_surf = big_font.render(winner_text, True, (255, 220, 90))
            screen.blit(w_surf, (WIDTH // 2 - w_surf.get_width() // 2, HEIGHT // 2 - 48))
            sub = font.render("Press [R] to fight again, or [M] to return to Menu", True, (200, 210, 225))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 16))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
