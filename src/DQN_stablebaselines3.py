import gymnasium as gym
from stable_baselines3 import DQN
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

vec_env = IcssimEnviroment()

log_dir = "retest/"
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = DQN("MlpPolicy", vec_env, verbose=2)
model.set_logger(logger)
model.learn(total_timesteps=1000, log_interval=1)
model.save("retest/dqn_icssim_1000_5")
print("done")