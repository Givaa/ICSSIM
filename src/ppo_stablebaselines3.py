import gymnasium as gym
from stable_baselines3 import PPO
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = IcssimEnviroment()

model = PPO("MlpPolicy", vec_env, verbose=1, ent_coef=0.1)
model.learn(total_timesteps=25000)
model.save("ppo_icssim_new")
print("done")

del model

model = PPO.load("ppo_icssim_new")

obs = vec_env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = vec_env.step(action)
    vec_env.render("human")