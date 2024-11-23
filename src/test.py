import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from stable_baselines3 import DQN
from icssim_enviroment import IcssimEnviroment

class DQN_causal(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_causal, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

NUM_EPISODES = 20
DQN_MODELS = [
    "retest/DQN_1000_1/dqn_icssim_1000_1.zip",
    "retest/DQN_1000_2/dqn_icssim_1000_2.zip",
    "retest/DQN_1000_3/dqn_icssim_1000_3.zip",
    "retest/DQN_1000_4/dqn_icssim_1000_4.zip",
    "retest/DQN_1000_5/dqn_icssim_1000_5.zip",
]
CAUSAL_MODELS = [
    "retest/DQN_causal_1000_1/DQN_causal_1000_1.pth",
    "retest/DQN_causal_1000_2/DQN_causal_1000_2.pth",
    "retest/DQN_causal_1000_3/DQN_causal_1000_3.pth",
    "retest/DQN_causal_1000_4/DQN_causal_1000_4.pth",
    "retest/DQN_causal_1000_5/DQN_causal_1000_5.pth",
    "retest/DQN_causal_1000_6/DQN_causal_1000_6.pth",
]
BASE_DIR = "test_results"
os.makedirs(BASE_DIR, exist_ok=True)

env = IcssimEnviroment()
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

def test_dqn_models():
    for model_path in DQN_MODELS:
        model_name = model_path.split('/')[-2]
        print(f"Test del modello DQN: {model_name}")
        model = DQN.load(model_path)
        model_results_dir = os.path.join(BASE_DIR, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        columns = ['Episodio', 'Timestep', 'Stato', 'Azione', 'Ricompensa', 'Stato successivo', 'Terminato']
        df = pd.DataFrame(columns=columns)

        for episode in range(NUM_EPISODES):
            obs, info = env.reset()
            for step in range(10):
                action, _states = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                df = pd.concat([df, pd.DataFrame({
                    'Episodio': [episode],
                    'Timestep': [step],
                    'Stato': [obs.tolist()],
                    'Azione': [action],
                    'Ricompensa': [reward],
                    'Stato successivo': [next_obs.tolist()],
                    'Terminato': [done],
                })])

                obs = next_obs
                if done:
                    break

        csv_path = os.path.join(model_results_dir, f"{model_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Risultati salvati per il modello DQN: {csv_path}")

def test_causal_models():
    for model_path in CAUSAL_MODELS:
        model_name = model_path.split('/')[-2]
        print(f"Test del modello causale: {model_name}")

        model = DQN_causal(n_observations, n_actions)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        model_results_dir = os.path.join(BASE_DIR, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        columns = ['Episodio', 'Timestep', 'Stato', 'Azione', 'Ricompensa', 'Stato successivo', 'Terminato']
        df = pd.DataFrame(columns=columns)

        for episode in range(NUM_EPISODES):
            obs, info = env.reset()
            for step in range(10):
                with torch.no_grad():
                    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_values = model(state_tensor)
                    action = torch.argmax(action_values).item()

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                df = pd.concat([df, pd.DataFrame({
                    'Episodio': [episode],
                    'Timestep': [step],
                    'Stato': [obs.tolist()],
                    'Azione': [action],
                    'Ricompensa': [reward],
                    'Stato successivo': [next_obs.tolist()],
                    'Terminato': [done],
                })])

                obs = next_obs
                if done:
                    break

        csv_path = os.path.join(model_results_dir, f"{model_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Risultati salvati per il modello causale: {csv_path}")

if __name__ == "__main__":
    #print("Inizio test per i modelli DQN...")
    #test_dqn_models()
    print("Inizio test per i modelli causali...")
    test_causal_models()
    print("Test completati. Risultati salvati.")

