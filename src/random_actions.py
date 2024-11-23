import csv
import numpy as np
from icssim_enviroment import IcssimEnviroment

# Numero massimo di timesteps
MAX_TIMESTEPS = 1500
MAX_EPISODES = 200  # Numero massimo di episodi completati

# File CSV di output
CSV_FILE = "retest/random_200_episodi_5/random_data_200_episodi_5.csv"
LOG_FILE = "retest/random_200_episodi_5/logs_200_episodi_5.csv"

# Ambiente
env = IcssimEnviroment()

# Inizializzazione dei file CSV
# File per i dettagli di ogni passo
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestep", "Episodio", "Stato", "Azione", "Ricompensa", "Stato successivo", "Terminato"])

# File per il tracking dei log dell'episodio
with open(LOG_FILE, mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Episodio", "Durata/Episodio", "Loss/Episodio", "Reward/Episodio"])

# Loop principale dell'agente
episode = 1
timestep = 0
while timestep < MAX_TIMESTEPS and episode <= MAX_EPISODES:  # Controlla anche il numero di episodi
    # Reset dell'ambiente all'inizio dell'episodio
    state, _ = env.reset()

    done = False
    episode_reward = 0
    episode_loss = 0
    episode_duration = 0

    while not done and timestep < MAX_TIMESTEPS:
        # Azione casuale
        action = env.action_space.sample()

        # Esegui un passo nell'ambiente
        next_state, reward, terminated, truncated, info = env.step(action)

        # Calcola una perdita fittizia (puoi sostituire con un vero modello ML)
        loss = np.random.random()  # Valore fittizio per la perdita

        # Salva i dettagli del passo nel file CSV
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestep + 1,
                episode,
                state.tolist(),
                action,
                reward,
                next_state.tolist(),
                terminated or truncated
            ])

        # Aggiorna i valori per l'episodio
        state = next_state
        episode_reward += reward
        episode_loss += loss
        episode_duration += 1
        timestep += 1

        done = terminated or truncated

    # Salva i log dell'episodio
    with open(LOG_FILE, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            episode,
            episode_duration,
            episode_loss / episode_duration,  # Media della perdita per timestep
            episode_reward
        ])

    # Incrementa l'episodio
    episode += 1

