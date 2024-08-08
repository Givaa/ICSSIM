import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from icssim_enviroment import IcssimEnviroment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from dowhy import CausalModel
import time

# Initialize the custom environment
env = IcssimEnviroment()

# Define column names for the dataset
columns = ['Actual_input_valve_status', 'Actual_input_valve_mode', 'Actual_tank_level_value',  'Actual_tank_level_min', 'Actual_tank_level_max', 'Actual_tank_output_valve_status', 'Actual_tank_output_valve_mode', 'Actual_tank_output_flow_value', 'Actual_belt_engine_status', 'Actual_belt_engine_mode', 'Actual_bottle_level_value', 'Actual_bottle_level_max', 'Actual_bottle_distance_to_filler_value', 'Action', 'Reward', 'New_input_valve_status', 'New_input_valve_mode', 'New_tank_level_value',  'New_tank_level_min', 'New_tank_level_max', 'New_tank_output_valve_status', 'New_tank_output_valve_mode', 'New_tank_output_flow_value', 'New_belt_engine_status', 'New_belt_engine_mode', 'New_bottle_level_value', 'New_bottle_level_max', 'New_bottle_distance_to_filler_value']
df = pd.DataFrame(columns=columns)
df.to_csv('data.csv', index=False)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()  # Interactive mode for plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a named tuple to store experiences in the replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """ A cyclic buffer of bounded size that holds the transitions observed recently """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the Deep Q-Network (DQN) model
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
    
# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get the number of actions and observations from the environment
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# Create policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
total_timesteps = 0
max_timesteps = 25000

# Function to select an action based on the causal model or random sampling (epsilon-greedy strategy)
def select_action(state, treatment_effects):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            best_action = None
            best_effect = float('-inf')

            for action, effect in treatment_effects.items():
                if effect is not None and effect > best_effect:
                    best_action = action
                    best_effect = effect

            if best_action is None:
                return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            else:
                return torch.tensor([[best_action]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# Lists to store episode durations and rewards for plotting
episode_durations = []
episode_rewards = []

# Function to plot episode durations
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result DQN')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Function to plot episode rewards
def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result DQN')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Function to optimize the model using the stored transitions
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Function to execute the causal model and estimate treatment effect
def execute_causal_model(action):
    data = pd.read_csv('data.csv')
    data.dropna(inplace=True)

    filtered_data = data[data['Action'] == action]

    if len(filtered_data) < 8:
        return None

    causal_model = CausalModel(
        data=data,
        treatment='Action',
        outcome='Reward',
        common_causes=['Actual_input_valve_status', 'Actual_input_valve_mode', 'Actual_tank_level_value',  'Actual_tank_level_min', 'Actual_tank_level_max', 'Actual_tank_output_valve_status', 'Actual_tank_output_valve_mode', 'Actual_tank_output_flow_value', 'Actual_belt_engine_status', 'Actual_belt_engine_mode', 'Actual_bottle_level_value', 'Actual_bottle_level_max', 'Actual_bottle_distance_to_filler_value', 'New_input_valve_status', 'New_input_valve_mode', 'New_tank_level_value',  'New_tank_level_min', 'New_tank_level_max', 'New_tank_output_valve_status', 'New_tank_output_valve_mode', 'New_tank_output_flow_value', 'New_belt_engine_status', 'New_belt_engine_mode', 'New_bottle_level_value', 'New_bottle_level_max', 'New_bottle_distance_to_filler_value']
    )

    identified_estimand = causal_model.identify_effect()

    try:
        estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        treatment_effect = estimate.value
    except np.linalg.LinAlgError:
        print("LinAlgError: SVD did not converge. Setting effect to None.")
        treatment_effect = None

    return treatment_effect

# Number of episodes depending on GPU availability
if torch.cuda.is_available():
    num_episodes = 1200
else:
    num_episodes = 100000000

start_time = time.time()

# Main training loop
for i_episode in range(num_episodes):
    if total_timesteps >= max_timesteps:
        break

    print(f"Episode number: {i_episode}")
    print(f"Current timesteps: {total_timesteps}")
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Execute causal model for each action
    treatment_effects = {}
    for action in range(env.action_space.n):
        treatment_effects[action] = execute_causal_model(action)

    for t in count():
        action = select_action(state, treatment_effects)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        total_timesteps += 1

        if total_timesteps >= max_timesteps:
            done = True

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        flat_list = [num for sublist in state.tolist() for num in sublist]

        fixed_state = [round(num, 3) if isinstance(num, (int, float)) else num for num in flat_list]
        fixed_new_state = [round(num, 3) if isinstance(num, (int, float)) else num for num in observation.tolist()]
        
        # Store experience in the dataframe
        df = pd.concat([df, pd.DataFrame({'Actual_input_valve_status': fixed_state[0], 
                                        'Actual_input_valve_mode': fixed_state[1], 
                                        'Actual_tank_level_value': fixed_state[2],  
                                        'Actual_tank_level_min': fixed_state[3], 
                                        'Actual_tank_level_max': fixed_state[4], 
                                        'Actual_tank_output_valve_status': fixed_state[5], 
                                        'Actual_tank_output_valve_mode': fixed_state[6], 
                                        'Actual_tank_output_flow_value': fixed_state[7], 
                                        'Actual_belt_engine_status': fixed_state[8], 
                                        'Actual_belt_engine_mode': fixed_state[9], 
                                        'Actual_bottle_level_value': fixed_state[10],
                                        'Actual_bottle_level_max': fixed_state[11], 
                                        'Actual_bottle_distance_to_filler_value': fixed_state[12], 
                                        'Action': action.item(),
                                        'Reward': reward.item(),
                                        'New_input_valve_status': fixed_new_state[0], 
                                        'New_input_valve_mode': fixed_new_state[1], 
                                        'New_tank_level_value': fixed_new_state[2],  
                                        'New_tank_level_min': fixed_new_state[3], 
                                        'New_tank_level_max': fixed_new_state[4], 
                                        'New_tank_output_valve_status': fixed_new_state[5], 
                                        'New_tank_output_valve_mode': fixed_new_state[6], 
                                        'New_tank_output_flow_value': fixed_new_state[7], 
                                        'New_belt_engine_status': fixed_new_state[8], 
                                        'New_belt_engine_mode': fixed_new_state[9], 
                                        'New_bottle_level_value': fixed_new_state[10], 
                                        'New_bottle_level_max': fixed_new_state[11], 
                                        'New_bottle_distance_to_filler_value': fixed_new_state[12],
                                        }, index=[0])])
        
        df.to_csv('data.csv', index=False)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()    
            break

end_time = time.time()
training_time = end_time - start_time

torch.save(policy_net.state_dict(), 'DQN_causal_2.pth')

print(f'Training complete in {training_time:.2f} seconds')
print(f'Total episodes: {i_episode}')
plot_durations(show_result=True)
plt.savefig('/src/result_duration_plot_DQN2.png')
plt.ioff()
plt.show()