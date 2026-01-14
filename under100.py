import pandas as pd
import ast

# 원본 CSV 파일 이름 (하드코딩)
original_filename = "~/asic/Dataset_Paral/signature/MiraiBotnet/MiraiBotnet_RARM_1_confidence_signature_train_ea15.csv"
output_filename = original_filename.replace(".csv", "_under100.csv")

# CSV 파일 읽기
df = pd.read_csv(original_filename)

# 'Verified_Signatures' 열에 있는 문자열을 실제 리스트로 파싱
signature_list = ast.literal_eval(df.loc[0, 'Verified_Signatures'])

# F1-Score 기준으로 정렬 (내림차순)
sorted_signatures = sorted(signature_list, key=lambda x: x['F1-Score'], reverse=True)

# 하위 100개 선택 (즉, 상위에서 100개 제외한 나머지)
under_100_signatures = sorted_signatures[100:]

# 기존의 DataFrame을 유지하면서 해당 시그니처로 대체
df.loc[0, 'Verified_Signatures'] = str(under_100_signatures)

# 새 파일로 저장
df.to_csv(output_filename, index=False)

print(f"Signatures ranked under top 100 saved to '{output_filename}'")
