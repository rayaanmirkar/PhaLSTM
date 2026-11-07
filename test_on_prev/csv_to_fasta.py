import pandas as pd
import os

data_splits = {
    "PhaLSTM/building_data/stratified_split/train_strat.csv": "train.fasta",
    "PhaLSTM/building_data/stratified_split/val_strat.csv": "val.fasta",
    "PhaLSTM/building_data/stratified_split/test_strat.csv": "test.fasta"
}


for csv_file, fasta_file in data_splits.items():
        if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
       
        with open(fasta_file, "w") as f:
                 for index, row in df.iterrows():
                        phage_id = str(row['phage_id']).strip()
                        sequence = str(row['proteome_sequence']).strip()

                        f.write(f">{phage_id}\n{sequence}\n")
