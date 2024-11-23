import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import csv
from icssim_enviroment import IcssimEnviroment

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

# Preparazione delle directory per i risultati
base_dir = "test_25k_timesteps/"
os.makedirs(base_dir, exist_ok=True)

# Numero di ripetizioni e episodi per ciascuna
num_repeats = 4
num_episodes = 10

# Esecuzione delle ripetizioni
for repeat in range(1, num_repeats + 1):
    episode_times = []
    episode_rewards = []
    episode_details = []  # Lista per salvare i dettagli degli episodi

    repeat_file = os.path.join(base_dir, f"dqn_causal_results_repeat_{repeat}.csv")
    print(f"Inizio ripetizione {repeat}...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        start_time = time.time()
        total_reward = 0
        timestep = 0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_tensor).argmax(dim=1).item()

            # Passo nell'ambiente
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Salva dettagli dell'episodio
            episode_details.append([episode, timestep, obs.tolist(), action, reward, next_obs.tolist(), terminated])
            obs = next_obs
            timestep += 1

            # Pausa per osservare il comportamento
            time.sleep(1)

            if terminated or truncated:
                end_time = time.time()
                episode_times.append(end_time - start_time)
                episode_rewards.append(total_reward)
                print(f"Ripetizione {repeat}, Episodio {episode + 1}/{num_episodes} completato.")
                break

    # Salvataggio dei risultati della ripetizione in un file CSV
    with open(repeat_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Scrivi intestazioni
        writer.writerow(["Episodio", "Timestep", "Stato", "Azione", "Ricompensa", "Stato successivo", "Terminato"])
        # Scrivi i dettagli degli episodi
        writer.writerows(episode_details)

    print(f"Ripetizione {repeat} completata. Risultati salvati in '{repeat_file}'.")

print(f"Tutti i test completati. Risultati salvati in '{base_dir}'.")
