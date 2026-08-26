import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('final.csv')

train_df, temp_df = train_test_split(df, test_size=0.40, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

train_df.to_csv('train_rand.csv', index=False)
val_df.to_csv('val_rand.csv', index=False)
test_df.to_csv('test_rand.csv', index=False)
