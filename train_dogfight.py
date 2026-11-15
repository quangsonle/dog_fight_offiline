# train_dogfight.py
import os
# Prevent OpenMP thread thrashing across parallel SubprocVecEnv workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from collections import deque
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from dogfight_core import FastNumpyPolicy
from dogfight_env import DogfightEnv

def export_sb3_to_npz(model, filepath):
    pi_net = model.policy.mlp_extractor.policy_net
    act_net = model.policy.action_net
    w1 = pi_net[0].weight.detach().cpu().numpy().T
    b1 = pi_net[0].bias.detach().cpu().numpy()
    w2 = pi_net[2].weight.detach().cpu().numpy().T
    b2 = pi_net[2].bias.detach().cpu().numpy()
    w3 = act_net.weight.detach().cpu().numpy().T
    b3 = act_net.bias.detach().cpu().numpy()
    np.savez(filepath, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

class LeagueEvalCallback(BaseCallback):
    """
    Every 150k steps:
    - Tests against 6 Hard, 6 540-kt BnZ, 6 Kiter, 6 Floor-Hugger, and 10 vs Current Champion.
    - Promotes a new Champion only with 0 self-crashes and strong Floor/Kiter/Champion wins.
    - Maintains a rolling pool of the latest 8 Champions for Self-Play.
    """
    def __init__(self, eval_freq=150_000, save_dir="models", verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.pool_dir = os.path.join(save_dir, "pool")
        os.makedirs(self.pool_dir, exist_ok=True)

        self.eval_hard = DogfightEnv(opponent_type="hard", max_steps=1600)
        self.eval_bnz = DogfightEnv(opponent_type="bnz", max_steps=1600)
        self.eval_kiter = DogfightEnv(opponent_type="kiter", max_steps=1600)
        self.eval_floor = DogfightEnv(opponent_type="floor", max_steps=1600)
        self.eval_champ = DogfightEnv(opponent_type="model", max_steps=1600)

        self.best_adapt_score = -1
        self.last_eval = 0
        self.champ_count = 0
        self.pool_paths = deque(maxlen=8)

        existing = sorted([f for f in os.listdir(self.pool_dir) if f.endswith(".npz")])
        for fname in existing[-8:]:
            self.pool_paths.append(os.path.join(self.pool_dir, fname))
            self.champ_count += 1

    def _run_batch(self, env, n_games):
        wins, gun_kills, self_crashes, hp_diff = 0, 0, 0, 0.0
        for i in range(n_games):
            obs, _ = env.reset(options={"fixed": (i % 4 == 0)})
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            if info["self_crashed"]:
                self_crashes += 1
            if info["win"] == 1:
                wins += 1
                if not info["opp_crashed"]:
                    gun_kills += 1
            hp_diff += (info["p1_hp"] - info["p2_hp"])
        return wins, gun_kills, self_crashes, hp_diff / n_games

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self.model.save(os.path.join(self.save_dir, "rl_latest"))

            w_h, gk_h, sc_h, hp_h = self._run_batch(self.eval_hard, 6)
            w_b, gk_b, sc_b, hp_b = self._run_batch(self.eval_bnz, 6)
            w_k, gk_k, sc_k, hp_k = self._run_batch(self.eval_kiter, 6)
            w_f, gk_f, sc_f, hp_f = self._run_batch(self.eval_floor, 6)

            bot_wins = w_h + w_b + w_k + w_f
            bot_gk = gk_h + gk_b + gk_k + gk_f
            total_sc = sc_h + sc_b + sc_k + sc_f

            champ_path_npz = os.path.join(self.save_dir, "rl_champion.npz")
            has_champ = os.path.exists(champ_path_npz)

            if has_champ:
                self.eval_champ.set_stage_config("model", [champ_path_npz])
                self.eval_champ.active_opp_model = FastNumpyPolicy(champ_path_npz)
                w_c, gk_c, sc_c, hp_c = self._run_batch(self.eval_champ, 10)
                total_sc += sc_c
            else:
                w_c, hp_c = 10, 0.0

            print(f"\n[LEAGUE EVAL] Step {self.num_timesteps:,} | "
                  f"Hard:{w_h}/6 BnZ:{w_b}/6 Kiter:{w_k}/6 Floor:{w_f}/6 (GK:{bot_gk}/24) | "
                  f"vsChamp:{w_c}/10 (HP:{hp_c:+.1f}) | SelfCrash:{total_sc}")

            promote = False
            adapt_score = w_f + w_k
            if not has_champ and w_f >= 5 and total_sc == 0 and bot_wins >= 18:
                promote = True
            elif has_champ and adapt_score > self.best_adapt_score and w_f >= 5 and total_sc == 0 and bot_wins >= 19:
                self.best_adapt_score = adapt_score
                promote = True
            elif has_champ and w_c >= 6 and w_k >= 5 and w_f >= 5 and total_sc == 0 and bot_wins >= 20:
                promote = True

            if promote:
                self.champ_count += 1
                self.model.save(os.path.join(self.save_dir, "rl_champion"))
                export_sb3_to_npz(self.model, champ_path_npz)

                snap_path = os.path.join(self.pool_dir, f"champ_{self.champ_count:04d}.npz")
                export_sb3_to_npz(self.model, snap_path)
                self.pool_paths.append(snap_path)

                if self.num_timesteps >= 1_000_000:
                    self.training_env.env_method("set_stage_config", "self_play", list(self.pool_paths))

                print(f"  >>> CHAMPION #{self.champ_count} PROMOTED! Saved models/rl_champion.zip")
        return True

def verify_numpy_export(model):
    test_obs = np.random.uniform(-1, 1, size=(28,)).astype(np.float32)
    export_sb3_to_npz(model, "models/_verify.npz")
    np_pol = FastNumpyPolicy("models/_verify.npz")
    a_sb3, _ = model.predict(test_obs, deterministic=True)
    a_np = np_pol.predict(test_obs, deterministic=True)
    assert list(a_sb3) == list(a_np), f"Export mismatch: {a_sb3} vs {a_np}"
    os.remove("models/_verify.npz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from models/rl_latest.zip")
    parser.add_argument("--adapt", action="store_true", help="Adapt Champion against Floor-Hugger & Kiter exploits")
    args = parser.parse_args()

    os.makedirs("models/pool", exist_ok=True)
    torch.set_num_threads(2)

    n_envs = 8
    env = make_vec_env(lambda: DogfightEnv(opponent_type="easy"), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256]))

    eval_cb = LeagueEvalCallback(eval_freq=150_000, save_dir="models")
    model = None

    try:
        if args.adapt and os.path.exists("models/rl_champion.zip"):
            print("\n============================================================")
            print(" ADAPTING CHAMPION TO DEFEAT FLOOR-BAIT & KITING EXPLOITS")
            print("============================================================")
            model = PPO.load("models/rl_champion", env=env, device="cpu")
            model.ent_coef = 0.002
            model.lr_schedule = lambda _: 1e-4
            env.env_method("set_stage_config", "self_play", list(eval_cb.pool_paths))
            model.learn(total_timesteps=20_000_000, callback=eval_cb, reset_num_timesteps=False)

        elif args.resume and os.path.exists("models/rl_latest.zip"):
            print("\n============================================================")
            print(" RESUMING SELF-PLAY LEAGUE FROM models/rl_latest.zip")
            print("============================================================")
            model = PPO.load("models/rl_latest", env=env, device="cpu")
            model.ent_coef = 0.0004
            model.lr_schedule = lambda _: 8e-5
            if len(eval_cb.pool_paths) > 0:
                env.env_method("set_stage_config", "self_play", list(eval_cb.pool_paths))
            else:
                env.env_method("set_stage_config", "mixed", [])
            model.learn(total_timesteps=50_000_000, callback=eval_cb, reset_num_timesteps=False)

        else:
            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=8,
                gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=policy_kwargs, device="cpu", verbose=1
            )
            verify_numpy_export(model)

            print("\n============================================================")
            print(" STAGE 1/3: Flight & Ballistic Gunnery (200k steps)")
            print("============================================================")
            model.learn(total_timesteps=200_000)
            model.save("models/rl_stage1")

            print("\n============================================================")
            print(" STAGE 2/3: Defeating Scripted Bots (800k steps)")
            print("============================================================")
            env.env_method("set_stage_config", "mixed", [])
            model.ent_coef = 0.002
            model.lr_schedule = lambda _: 2e-4
            model.learn(total_timesteps=800_000, callback=eval_cb, reset_num_timesteps=False)
            model.save("models/rl_vs_bots")

            print("\n============================================================")
            print(" STAGE 3/3: Long-Horizon Self-Play League (Press Ctrl+C anytime)")
            print("============================================================")
            if len(eval_cb.pool_paths) == 0:
                export_sb3_to_npz(model, "models/rl_champion.npz")
                export_sb3_to_npz(model, "models/pool/champ_0001.npz")
                eval_cb.pool_paths.append("models/pool/champ_0001.npz")
                model.save("models/rl_champion")

            env.env_method("set_stage_config", "self_play", list(eval_cb.pool_paths))
            model.ent_coef = 0.0004
            model.lr_schedule = lambda _: 8e-5
            model.learn(total_timesteps=50_000_000, callback=eval_cb, reset_num_timesteps=False)

    except KeyboardInterrupt:
        if model is not None:
            print("\n[STOPPED BY USER] Saving latest checkpoint safely...")
            model.save("models/rl_latest")
            print("Saved models/rl_latest.zip!")
    finally:
        env.close()

if __name__ == "__main__":
    main()
