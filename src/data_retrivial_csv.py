# import networkx as nx
# from dowhy import CausalModel
# import pandas as pd
# from Configs import TAG
# from icssim_enviroment import IcssimEnviroment 
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# env = IcssimEnviroment()

# # Inizializza un DataFrame vuoto
# data = pd.DataFrame()

# # Numero di righe che desideri nel dataset
# num_righe = 1000

# for _ in range(num_righe):

#     valore_simulatore = env.get_real_time_data()
#     # I valori delle variabili
#     valvola_ingresso_stato = valore_simulatore[0]
#     valvola_ingresso_modalità = valore_simulatore[1]
#     livello_serbatoio = valore_simulatore[2]
#     livello_serbatoio_min = valore_simulatore[3]
#     livello_serbatoio_max = valore_simulatore[4]
#     valvola_uscita_stato = valore_simulatore[5]
#     valvola_uscita_modalità = valore_simulatore[6]
#     flusso_valvola_uscita = valore_simulatore[7]
#     motore_nastro_stato = valore_simulatore[8]
#     motore_nastro_modalità = valore_simulatore[9]
#     livello_bottiglia = valore_simulatore[10]
#     livello_max_bottiglia = valore_simulatore[11]
#     distanza_bottiglia_riempitore = valore_simulatore[12]

#     time.sleep(0.5)

#     # Aggiungi una nuova riga al DataFrame
#     nuova_riga = pd.DataFrame({
#         'valvola_ingresso_stato': [valvola_ingresso_stato],
#         'valvola_ingresso_modalità': [valvola_ingresso_modalità],
#         'livello_serbatoio': [livello_serbatoio],
#         'livello_serbatoio_min': [livello_serbatoio_min],
#         'livello_serbatoio_max': [livello_serbatoio_max],
#         'valvola_uscita_stato': [valvola_uscita_stato],
#         'valvola_uscita_modalità': [valvola_uscita_modalità],
#         'flusso_valvola_uscita': [flusso_valvola_uscita],
#         'motore_nastro_stato': [motore_nastro_stato],
#         'motore_nastro_modalità': [motore_nastro_modalità],
#         'livello_bottiglia': [livello_bottiglia],
#         'livello_max_bottiglia': [livello_max_bottiglia],
#         'distanza_bottiglia_riempitore': [distanza_bottiglia_riempitore]
#     })

#     # Concatena la nuova riga al DataFrame esistente
#     data = pd.concat([data, nuova_riga], ignore_index=True)

# # Visualizzazione delle prime righe del DataFrame
# print(data.head())

# # Salva il DataFrame in un file CSV
# data.to_csv('data.csv', index=True)

import gymnasium as gym
import pandas as pd
from icssim_enviroment import IcssimEnviroment

# Crea l'ambiente
env = IcssimEnviroment()

# Inizializza l'ambiente e ottieni lo stato iniziale
stato, info = env.reset()

# Crea un DataFrame vuoto per memorizzare i dati
dati = pd.DataFrame(columns=['stato', 'azione', 'ricompensa', 'nuovo_stato', 'MITM', 'DDos', 'Nmap', 'Command Injection'])

for iterazione in range(10):  # Sostituisci con il numero di passaggi che desideri
    
    print("Iterazione numero: " + str(iterazione))

    # Scegli un'azione
    azione = env.action_space.sample()  # Sostituisci con la tua politica se ne hai una

    ddos = 0
    command_injection = 0
    mitm = 0
    nmap = 0

    if azione == 2 or azione == 3 or azione == 4 or azione == 5 or azione == 6 or azione == 7 or azione == 9 or azione == 10:
        command_injection = 1
    elif azione == 0:
        nmap = 1
    elif azione == 1 or azione == 8:
        ddos = 1
    else:
        mitm = 1

    # Esegui l'azione e ottieni il nuovo stato, la ricompensa e altre informazioni
    nuovo_stato, ricompensa, terminated, truncated, info = env.step(azione)

    # Crea un DataFrame temporaneo con i dati dell'iterazione corrente
    dati_iterazione = pd.DataFrame({
        'stato': [stato],
        'azione': [azione],
        'ricompensa': [ricompensa],
        'nuovo_stato': [nuovo_stato],
        'MITM': [mitm],
        'DDos': [ddos],
        'Nmap': [nmap],
        'Command Injection': [command_injection]
    })

    # Concatena il DataFrame temporaneo con il DataFrame principale
    dati = pd.concat([dati, dati_iterazione], ignore_index=True)

    # Aggiorna lo stato
    stato = nuovo_stato

    if terminated or truncated:
        # Se l'episodio è finito, resetta l'ambiente
        stato = env.reset()

# Salva i dati in un file CSV
dati.to_csv('dati.csv', index=False)
