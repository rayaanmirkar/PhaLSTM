import pandas as pd

df = pd.read_csv("final_data_training.csv") 
unique_genera = df['genus'].dropna().sample(frac=1, random_state=42).unique()

train_end = int(len(unique_genera) * 0.8)
val_end = int(len(unique_genera) * 0.9)

train_genera = unique_genera[:train_end]
val_genera = unique_genera[train_end:val_end]
test_genera = unique_genera[val_end:]

train_df = df[df['genus'].isin(train_genera)]
val_df = df[df['genus'].isin(val_genera)]
test_df = df[df['genus'].isin(test_genera)]

train_df.to_csv("new_train_split.csv", index=False)
val_df.to_csv("new_val_split.csv", index=False)
test_df.to_csv("new_test_split.csv", index=False)