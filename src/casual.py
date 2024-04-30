import gymnasium as gym
from icssim_enviroment import IcssimEnviroment
import matplotlib.pyplot as plt
import time

env = IcssimEnviroment()

observation, info = env.reset()

num_episodes = 10

episode_times = []
episode_rewards = []

for episode in range(num_episodes):
    observation, info = env.reset()

    terminated = False
    truncated = False
    start_time = time.time()
    total_reward = 0
    
    while not terminated and not truncated:

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward

    end_time = time.time()
    episode_time = end_time - start_time
    
    episode_times.append(episode_time)
    episode_rewards.append(total_reward)

    print(f"Episodio {episode + 1}: Tempo impiegato = {episode_time}, Reward totale = {total_reward}")

env.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_times, marker='o', linestyle='-')
plt.title('Tempo impiegato per episodio')
plt.xlabel('Episodio')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.show()
plt.savefig('1.png')

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', linestyle='-')
plt.title('Reward ottenuto per episodio')
plt.xlabel('Episodio')
plt.ylabel('Reward')
plt.grid(True)
plt.show()
plt.savefig('2.png')


