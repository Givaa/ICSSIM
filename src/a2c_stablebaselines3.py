from stable_baselines3 import A2C
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv

# Parallel environments
vec_env = IcssimEnviroment()

model = A2C("MlpPolicy", vec_env, verbose=1, device="cpu", ent_coef=0.1)
model.learn(total_timesteps=10000)
model.save("a2c_icssim")
print("done")

del model # remove to demonstrate saving and loading

model = A2C.load("models/a2c_icssim")

obs = vec_env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    # vec_env.render("human")