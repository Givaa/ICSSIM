import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl import SimulatoreAmbiente

# Crea l'ambiente
env = DummyVecEnv([lambda: SimulatoreAmbiente()])

# Crea il modello PPO
model = PPO("MlpPolicy", env, verbose=1)

# Addestra il modello per 100 episodi
model.learn(total_timesteps=100, progress_bar=True)

# Valuta il modello dopo l'addestramento
mean_reward, std_reward = model.evaluate(env, n_eval_episodes=10)

print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")