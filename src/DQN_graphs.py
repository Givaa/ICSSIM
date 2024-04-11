import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
from icssim_enviroment import IcssimEnviroment
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import time

# Creare un'istanza dell'ambiente
env = IcssimEnviroment()

# Caricare il modello DQN addestrato
model = A2C.load("a2c_icssim")

# Liste per memorizzare i tempi impiegati per episodio e il numero di episodi
episode_times = []
episode_rewards = []

# Numero totale di episodi
num_episodes = 10

for _ in range(num_episodes):
    # Reset dell'ambiente per iniziare un nuovo episodio
    obs, info = env.reset()

    # Eseguire l'episodio fino alla fine
    start_time = time.time()
    total_reward = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1)
        total_reward += reward

        # Se l'episodio è terminato, registrare il tempo impiegato e il reward totale
        if terminated == True or truncated == True:
            end_time = time.time()
            episode_times.append(end_time - start_time)
            episode_rewards.append(total_reward)
            break

# Grafico del tempo impiegato per episodio
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_times, marker='o', linestyle='-')
plt.title('Tempo impiegato per episodio')
plt.xlabel('Episodio')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.show()
plt.savefig('DQN_stable_time.png')

# Grafico del reward ottenuto per episodio
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', linestyle='-')
plt.title('Reward ottenuto per episodio')
plt.xlabel('Episodio')
plt.ylabel('Reward')
plt.grid(True)
plt.show()
plt.savefig('DQN_stable_reward.png')
