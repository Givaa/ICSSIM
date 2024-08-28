import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from icssim_enviroment import IcssimEnviroment
import time

# Definizione dell'architettura del modello DQN
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Inizializzazione dell'ambiente
env = IcssimEnviroment()
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# Caricamento del modello e dei pesi
model = DQN(n_observations, n_actions)
model.load_state_dict(torch.load('modelli_nuovi/DQN_causal.pth'))
model.eval()

# Liste per memorizzare i tempi e i reward
episode_times = []
episode_rewards = []

num_episodes = 10

# Esecuzione degli episodi
for _ in range(num_episodes):
    obs, info = env.reset()
    start_time = time.time()
    total_reward = 0
    
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_tensor).argmax(dim=1).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        time.sleep(1)

        if terminated or truncated:
            end_time = time.time()
            episode_times.append(end_time - start_time)
            episode_rewards.append(total_reward)
            break

# Plot dei tempi per episodio
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_times, marker='o', linestyle='-')
plt.title('Tempo impiegato per episodio')
plt.xlabel('Episodio')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.savefig('modelli_nuovi/PyTorch_stable_time.png')
plt.show()

# Plot dei reward per episodio
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', linestyle='-')
plt.title('Reward ottenuto per episodio')
plt.xlabel('Episodio')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig('modelli_nuovi/PyTorch_stable_reward.png')
plt.show()

