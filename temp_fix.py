file_path = "Modules/Jaccard_Elbow_Method.py"
lines = []
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_summary_block = False

# Lines to completely remove
lines_to_remove = list(range(57, 89))  # Corresponds to lines 58-89 in original file
lines_to_remove += list(range(1478, 1484)) # Corresponds to lines 1479-1484

# Keywords to remove lines
keywords_to_remove = ["import json"]

with open(file_path, 'w', encoding='utf-8') as f:
    for i, line in enumerate(lines):
        # Skip specific line ranges
        if (57 <= i + 1 <= 89) or (1478 <= i + 1 <= 1484):
            continue
        
        # Skip lines with keywords
        if any(keyword in line for keyword in keywords_to_remove):
            continue
            
        f.write(line)

print("File cleanup complete.")
