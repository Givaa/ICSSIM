import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

def tensorboard_to_csv(logdir, csv_output_path):
    """
    Converte i log di TensorBoard (file .tfevents) in un file CSV.

    :param logdir: Directory contenente i file TensorBoard.
    :param csv_output_path: Percorso dove salvare il file CSV generato.
    """
    tfevents_file = None
    for root, _, files in os.walk(logdir):
        for file in files:
            if "events.out.tfevents" in file:
                tfevents_file = os.path.join(root, file)
                break
    if tfevents_file is None:
        raise FileNotFoundError(f"Nessun file TensorBoard trovato nella directory {logdir}")
    
    print(f"File TensorBoard trovato: {tfevents_file}")
    
    data = []
    for event in summary_iterator(tfevents_file):
        for value in event.summary.value:
            if value.HasField('simple_value'): 
                data.append({
                    "step": event.step,
                    "tag": value.tag,
                    "value": value.simple_value
                })
    
    if not data:
        raise ValueError("Nessun dato scalare trovato nei log di TensorBoard.")
    
    df = pd.DataFrame(data)
    df_pivoted = df.pivot(index="step", columns="tag", values="value")
    
    # Salva il dataframe come CSV prima di modificarlo
    df_pivoted.to_csv(csv_output_path)
    print(f"Log di TensorBoard convertito in CSV: {csv_output_path}")
    
    # Modifica il CSV come richiesto
    modify_csv(csv_output_path)

def modify_csv(csv_path):
    """
    Modifica il file CSV sottraendo 1 dal valore della seconda colonna e
    sottraendo il risultato dal valore della quarta colonna.
    
    :param csv_path: Percorso del file CSV da modificare.
    """
    
    df = pd.read_csv(csv_path)
    if df.shape[1] < 4:
        raise ValueError("Il CSV deve contenere almeno 4 colonne.")

    for index, row in df.iterrows():
        second_col_value = row.iloc[1]  
        fourth_col_value = row.iloc[3] 
        new_value = fourth_col_value - (second_col_value - 1)
        
        df.at[index, df.columns[3]] = new_value

    df.to_csv(csv_path, index=False)
    print(f"File CSV modificato salvato come: {csv_path}")

if __name__ == "__main__":
    logdir = "retest/DQN_causal_1000_6" 
    csv_output_path = "retest/DQN_causal_1000_6/DQN_causal_1000_6.csv"
    
    try:
        tensorboard_to_csv(logdir, csv_output_path)
    except Exception as e:
        print(f"Errore: {e}")