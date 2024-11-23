from stable_baselines3 import A2C
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

vec_env = IcssimEnviroment()

log_dir = "./retest/"
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = A2C("MlpPolicy", vec_env, verbose=2, device="cpu", ent_coef=0.1)
model.set_logger(logger)
model.learn(total_timesteps=1000, log_interval=1)
model.save("retest/a2c_icssim")
print("done")
