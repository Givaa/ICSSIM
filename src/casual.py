import gymnasium as gym
from icssim_enviroment import IcssimEnviroment
import matplotlib.pyplot as plt
import time

# Creare un'istanza dell'ambiente
env = IcssimEnviroment()

# Reset dell'ambiente per iniziare una nuova episodio
observation, info = env.reset()

# Numero totale di episodi
num_episodes = 10

# Liste per memorizzare i tempi impiegati per episodio e il numero di episodi
episode_times = []
episode_rewards = []

for episode in range(num_episodes):
    # Reset dell'ambiente per iniziare un nuovo episodio
    observation, info = env.reset()

    # Eseguire l'episodio fino alla fine
    terminated = False
    truncated = False
    start_time = time.time()
    total_reward = 0
    
    while not terminated and not truncated:
        # Selezionare un'azione casuale dall'ambiente
        action = env.action_space.sample()

        # Eseguire l'azione nell'ambiente e ottenere le nuove osservazioni e il reward
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Aggiornare il reward totale
        total_reward += reward

    # Calcolare il tempo impiegato per l'episodio
    end_time = time.time()
    episode_time = end_time - start_time
    
    # Aggiungere il tempo impiegato e il reward totale alla lista
    episode_times.append(episode_time)
    episode_rewards.append(total_reward)

    # Stampare le informazioni sull'episodio
    print(f"Episodio {episode + 1}: Tempo impiegato = {episode_time}, Reward totale = {total_reward}")

# Chiudere l'ambiente
env.close()

# Grafico del tempo impiegato per episodio
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_times, marker='o', linestyle='-')
plt.title('Tempo impiegato per episodio')
plt.xlabel('Episodio')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.show()
plt.savefig('1.png')

# Grafico del reward ottenuto per episodio
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', linestyle='-')
plt.title('Reward ottenuto per episodio')
plt.xlabel('Episodio')
plt.ylabel('Reward')
plt.grid(True)
plt.show()
plt.savefig('2.png')


