import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'class_label': ['A']*250 + ['B']*14 + ['C']*35 + ['D']*14 + ['E']*10 + ['F']*42 + ['G']*250 + ['H']*12 + ['I']*100 + ['J']*12 + ['K']*20 + ['L']*6 + ['M']*250 + ['N']*12
})

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['class_label'])
print("First split successful.")

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['class_label'])
print("Second split successful.")
