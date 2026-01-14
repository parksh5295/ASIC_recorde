import sys
import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from utils.apply_labeling import apply_labeling_logic


# -------- CONFIG --------
input_csv = "C:\\ASIC_excute\\Dataset\\load_dataset\\training-flow.csv"
output_csv = "C:\\ASIC_excute\\Dataset\\mini_dataset\\CICIoT2023_attack10.csv"
file_type = "CICIoT2023"
n_rows = 10
# ------------------------

# Load dataset
df = pd.read_csv(input_csv)

# Apply unified labeling logic
df = apply_labeling_logic(df, file_type)

# Filter only attack data (label == 1)
attack_df = df[df['label'] == 1]

# Maintain original order and take the first n_rows
attack_head = attack_df.head(n_rows)

# Save to CSV (include column names)
attack_head.to_csv(output_csv, index=False)

print(f"[OK] Saved {len(attack_head)} attack rows to '{output_csv}'")
