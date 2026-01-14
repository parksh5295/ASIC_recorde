# python file to generate chunks for feature checking

import pandas as pd

# --- New Efficient Version ---
# This version reads only the first 8 data rows from the source file, which is much faster.

output_file = '../Dataset/chunk_dataset/IoTID20_dataset.csv'
input_file = "../Dataset/load_dataset/IoTID20/IoTID20.csv"

# The 'nrows' parameter tells pandas to stop reading after 8 rows, avoiding loading the entire file.
print(f"Reading the first 8 data rows from '{input_file}'...")
df = pd.read_csv(input_file, nrows=8)

# The result is saved to the output file just once.
print(f"Saving 8 rows to '{output_file}'...")
df.to_csv(output_file, header=True, index=False)

print(f"Successfully created a chunk with 8 rows at '{output_file}'")


'''

# --- Old Inefficient Version (Commented Out) ---
# The code below was very slow because it iterated through the ENTIRE source file
# in small chunks, repeatedly overwriting the output file in every loop.

chunk_size = 10
# output_file = '../Dataset/chunk_dataset/MiraiBotnet_chunks.csv'
# output_file = '../Dataset/chunk_dataset/Kitsune_chunks.csv'
# output_file = '../Dataset/chunk_dataset/netML_dataset.csv'
# output_file = '../Dataset/chunk_dataset/CICIDS2017_dataset.csv'
# output_file = '../Dataset/chunk_dataset/IoTID20_dataset.csv'
output_file = '../Dataset/chunk_dataset/CICIoT2023_dataset.csv'

# Overwrite only 6 lines in each batch
# for i, chunk in enumerate(pd.read_csv('../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv', chunksize=chunk_size)):
# for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv", chunksize=chunk_size)):
# for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/netML/netML_dataset.csv", chunksize=chunk_size)):
# for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/CICIDS2017/MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", chunksize=chunk_size)):
# for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/IoTID20/IoTID20.csv", chunksize=chunk_size)):
for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/CICIoT2023/training-flow.csv", chunksize=chunk_size)):
    first_6_rows = chunk.head(8)
    
    first_6_rows.to_csv(output_file, mode='w', header=True, index=False)    # Save as overwrite on redo

'''