import gymnasium as gym
from stable_baselines3 import DQN
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = IcssimEnviroment()

model = DQN("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000, log_interval=4)
model.save("dqn_icssim")
print("done")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_icssim")

obs, info = vec_env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    if terminated or truncated:
        obs, info = vec_env.reset()
