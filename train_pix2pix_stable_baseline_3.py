from config import (
    LOAD_MODEL,
    CHECKPOINT_PATH,
    SAVE_HISTORY_VAL_DIR,
)

from env import LambdaTuningEnv, EarlyStopCallback, VecNormalizeHistory


import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def save_vec_normalize_history(array_list, filename="output.csv", columns=[]):
    """Save observation history from vectorized environment normalization"""
    flat = [arr.reshape(-1) for arr in array_list]
    stacked = np.vstack(flat)
    num_cols = stacked.shape[1]
    df = pd.DataFrame(stacked, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows with {num_cols} columns to {filename}")


# Setup environment
env = DummyVecEnv([LambdaTuningEnv])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=20.0)
env = VecNormalizeHistory(env)

# Load or create model
if LOAD_MODEL:
    print(f"load model at {CHECKPOINT_PATH}")
    model = PPO.load(
        CHECKPOINT_PATH,
        env=env,
        verbose=1,
        tensorboard_log="./logs",
    )
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs",
        n_steps=16,
        n_epochs=4,
        use_sde=True,
        sde_sample_freq=4,
        ent_coef=0,
    )

# Setup callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=64, save_path="./checkpoints/", name_prefix="lambda_tuner"
)

early_stopping_callback = EarlyStopCallback(patience=500)

callback = CallbackList([checkpoint_callback])

# Train model
model.learn(total_timesteps=5000, reset_num_timesteps=True, callback=callback)

# Save history
save_vec_normalize_history(
    env.obs_history,
    filename=f"{SAVE_HISTORY_VAL_DIR}/obs_normalize_history.csv",
    columns=[
        "l1_lambda",
        "perc_lambda",
        "style_lambda",
        "l1_loss",
        "perc_loss",
        "style_loss",
        "fid_score",
        "lpips_score",
    ],
)

save_vec_normalize_history(
    env.reward_history,
    filename=f"{SAVE_HISTORY_VAL_DIR}/reward_normalize_history.csv",
    columns=[
        "reward",
    ],
)
