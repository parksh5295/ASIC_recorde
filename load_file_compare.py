import csv


def compare_csv(file1, file2):
    differences = []
    
    with open(file1, newline='', encoding='utf-8') as f1, \
         open(file2, newline='', encoding='utf-8') as f2:
        
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        
        for row_idx, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            for col_idx, (val1, val2) in enumerate(zip(row1, row2), start=1):
                if val1 != val2:
                    differences.append((row_idx, col_idx, val1, val2))
    
    if not differences:
        print("✅ 두 CSV 파일의 모든 요소가 동일합니다.")
    else:
        print(f"❌ {len(differences)} 개의 차이가 발견되었습니다:")
        for diff in differences:
            row, col, v1, v2 = diff
            print(f"  - Row {row}, Col {col}: {v1} != {v2}")


# 사용 예시
compare_csv("../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv", "../Dataset/load_dataset/MiraiBotnet/MiraiBotnet_powershell.csv")
