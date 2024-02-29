from rl import SimulatoreAmbiente
import random
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Inizializza l'agente DQN con i parametri specificati
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Memorizza le esperienze passate
        self.gamma = 0.95  # Fattore di sconto per le ricompense future
        self.epsilon = 1.0  # Esplorazione iniziale (100% di probabilità di esplorare)
        self.epsilon_decay = 0.995  # Riduzione dell'esplorazione ad ogni passo
        self.epsilon_min = 0.01  # Esplorazione minima (1%)
        self.learning_rate = 0.001  # Tasso di apprendimento per l'ottimizzatore
        self.model = self._build_model()  # Costruisce il modello della rete neurale
        self.available_actions = list(range(action_size))  # Lista delle azioni disponibili

    def _build_model(self):
        # Costruisce il modello della rete neurale con architettura specifica
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Memorizza un'esperienza nella memoria dell'agente
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # L'agente decide l'azione da intraprendere in base all'attuale stato
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Azione casuale con probabilità epsilon
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Azione migliore secondo il modello

    def replay(self, batch_size):
        # Esegue l'aggiornamento della rete neurale attraverso l'apprendimento dall'esperienza
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            print(next_state)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Riduci l'esplorazione ad ogni passo

# Inizializza l'ambiente di simulazione
env = SimulatoreAmbiente()

# Inizializza l'agente DQN
state_size = len(env.observation_space.sample())
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Parametri di addestramento
batch_size = 32
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, len(state)])
    total_reward = 0

    action = 0 # Esegue sempre una scansione nmap all'inizio
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, len(next_state)])
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    total_reward += reward

    for time in range(500):  # Limite di passi per episodio
        action = agent.act(state)  # L'agente decide l'azione da intraprendere
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, len(next_state)])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print("Episodio: {}/{}, Ricompensa: {}".format(episode, episodes, total_reward))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)  # Esegui il replay per apprendere dall'esperienza