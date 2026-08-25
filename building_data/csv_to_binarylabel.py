import pandas as pd

df = pd.read_csv("C:/Users/raypi/coding/PhaLSTM/building_data/final_data_training.csv") 

df["Binary lifestyle"] = df["Temperate (empirical)"].str.lower().map({"yes" : 1, "no": 0})

df.to_csv("final.csv", index=False)

print("successfully created")
