import gymnasium as gym
from stable_baselines3 import PPO
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

vec_env = IcssimEnviroment()

log_dir = "./retest/"
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = PPO("MlpPolicy", vec_env, verbose=2, ent_coef=0.1)
model.set_logger(logger)
model.learn(total_timesteps=500, log_interval=1)
model.save("retest/ppo_icssim")
print("done")