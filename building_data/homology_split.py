import pandas as pd

df = pd.read_csv("C:/Users/raypi/coding/PhaLSTM/building_data/final.csv") 
unique_genera = pd.Series(df['Host genus'].dropna().unique()).sample(frac=1, random_state=42).to_numpy()

train_end = int(len(unique_genera) * 0.8)
val_end = int(len(unique_genera) * 0.9)

train_genera = unique_genera[:train_end]
val_genera = unique_genera[train_end:val_end]
test_genera = unique_genera[val_end:]

df[df['Host genus'].isin(train_genera)].to_csv("train_split.csv", index=False)
df[df['Host genus'].isin(val_genera)].to_csv("val_split.csv", index=False)
df[df['Host genus'].isin(test_genera)].to_csv("test_split.csv", index=False)