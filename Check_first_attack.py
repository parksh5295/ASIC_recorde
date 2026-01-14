import pandas as pd

# Import Kitsune CSV
#df = pd.read_csv("../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv")
#df = pd.read_csv("../Dataset/load_dataset/CICIoT2023/training-flow.csv")
df = pd.read_csv("C:\\ASIC_excute\\Dataset\\load_dataset\\DARPA98_train.csv")

# Match the label column name (now 'Label' → 'label')
#df['label'] = df['attack_flag']
df['label'] = df['Flag']

# Sort by time (can be omitted if already sorted)
# df = df.sort_values("Timestamp").reset_index(drop=True)

# Find the first attack line
first_attack_idx = df.index[df['label'] == 1]

if len(first_attack_idx) > 0:
    row_num = first_attack_idx[0]           # 0-based index
    print("First attack row (0-based):", row_num)
    print("First attack row (1-based):", row_num + 1)
    print(df.iloc[row_num])                 # Print the entire row
else:
    print("No attack found.")
