import os
import csv
from stable_baselines3 import DQN
from icssim_enviroment import IcssimEnviroment

# Carica il modello pre-addestrato
MODEL_PATH = "modelli_nuovi/dqn_icssim.zip"
model = DQN.load(MODEL_PATH)

# Cartella per i risultati
base_dir = "test_25k_timesteps/DQN"
os.makedirs(base_dir, exist_ok=True)

# Parametri di test
num_repeats = 4
TEST_EPISODES = 10

# Ambiente di test
env = IcssimEnviroment()

# Esegui le ripetizioni
for repeat in range(1, num_repeats + 1):
    repeat_file = os.path.join(base_dir, f"dqn_icssim_results_repeat_{repeat}.csv")
    print(f"Inizio ripetizione {repeat}...")

    # Scrivi le intestazioni per ogni file CSV
    with open(repeat_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episodio", "Timestep", "Stato", "Azione", "Ricompensa", "Stato successivo", "Terminato"])

    # Esegui gli episodi per questa ripetizione
    for episode in range(TEST_EPISODES):
        state, _ = env.reset()
        terminated = False
        truncated = False
        timestep = 0

        while not (terminated or truncated):
            # Previsione del modello
            action, _states = model.predict(state, deterministic=True)

            # Step nell'ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Salva i risultati nel file CSV
            with open(repeat_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    episode, timestep,
                    state.tolist(),
                    action, reward,
                    next_state.tolist(),
                    terminated
                ])

            # Avanza allo stato successivo
            state = next_state
            timestep += 1

        print(f"Ripetizione {repeat}, Episodio {episode + 1}/{TEST_EPISODES} completato.")

    print(f"Ripetizione {repeat} completata. Risultati salvati in '{repeat_file}'.")

print(f"Tutti i test completati. Risultati salvati in '{base_dir}'.")
