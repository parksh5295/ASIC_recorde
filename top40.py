import pandas as pd
import ast

# 원본 CSV 파일 이름 (하드코딩)
original_filename = r"D:\AutoSigGen_withData\Dataset_Paral\signature\NSL-KDD\NSL-KDD_RARM_1_confidence_signature_train_ea15.csv"
output_filename = original_filename.replace(".csv", "_top40.csv")

# CSV 파일 읽기
df = pd.read_csv(original_filename)

# 'Verified_Signatures' 열에 있는 문자열을 실제 리스트로 파싱
signature_list = ast.literal_eval(df.loc[0, 'Verified_Signatures'])

# F1-Score 기준으로 정렬 (내림차순)
sorted_signatures = sorted(signature_list, key=lambda x: x['F1-Score'], reverse=True)

# 상위 40개만 선택
top_40_signatures = sorted_signatures[:40]

# 기존의 DataFrame을 유지하면서 상위 40개로 'Verified_Signatures'만 대체
df.loc[0, 'Verified_Signatures'] = str(top_40_signatures)

# 새 파일로 저장
df.to_csv(output_filename, index=False)

print(f"Top 40 signatures saved to '{output_filename}'")
