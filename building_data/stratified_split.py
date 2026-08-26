import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("final.csv")

counts = df["Host genus"].value_counts()
rare_mask = df["Host genus"].isin(counts[counts < 4].index)

df_rare = df[rare_mask].copy()
df_strat = df[~rare_mask].copy()

train_strat, temp_strat = train_test_split(
    df_strat,
    test_size=0.40,
    stratify=df_strat["Host genus"],
    random_state=42
)

val, test = train_test_split(
    temp_strat,
    test_size=0.50,
    stratify=temp_strat["Host genus"],
    random_state=42
)

train = pd.concat([train_strat, df_rare], ignore_index=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

train.to_csv("train_strat.csv", index=False)
val.to_csv("val_strat.csv", index=False)
test.to_csv("test_strat.csv", index=False)