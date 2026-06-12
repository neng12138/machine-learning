import pandas as pd
import numpy as np
from scipy import stats

input_file = r'./data clean Terklasifikasi  baru 22 juli 2015 all.csv'
output_file = r'./cleaned_data.csv'

print("=== 数据预处理开始 ===")

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

data_lines = []
for line in lines:
    line = line.strip()
    if line and not line.startswith('@'):
        data_lines.append(line)

columns = ['YEAR', 'SEX', 'AGE_CLASS', 'MARRYSTATUS', 'EDUCATIONLEVEL', 
           'LIVEPERSONNUM', 'LIVE_CLASS', 'WORKYEAR_CLASS', 'INCOME_CLASS', 
           'BUSINESSTYPE', 'EMPLOYMENTSTYPE', 'RESULT']

df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)

for col in columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['RESULT'] = df['RESULT'].astype(str)

print(f"原始数据行数: {len(df)}")

original_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = original_rows - len(df)
print(f"移除重复行: {duplicates_removed}")

missing_values = df.isnull().sum()
print("\n缺失值统计:")
print(missing_values)

print("\n各字段描述统计:")
print(df.describe())

print("\n各字段唯一值:")
for col in columns[:-1]:
    unique_vals = sorted(df[col].dropna().unique())
    print(f"{col}: {unique_vals[:10]}... (共{len(unique_vals)}个唯一值)")

print("\nRESULT分布:")
print(df['RESULT'].value_counts())

z_scores = np.abs(stats.zscore(df.drop('RESULT', axis=1), nan_policy='omit'))
outlier_mask = (z_scores > 3).any(axis=1)
outliers_count = outlier_mask.sum()
print(f"\nZ-score异常值检测: {outliers_count} 行")

print("\n=== 异常值截断处理 ===")
print(f"LIVEPERSONNUM处理前统计:")
print(f"  最小值: {df['LIVEPERSONNUM'].min()}")
print(f"  最大值: {df['LIVEPERSONNUM'].max()}")
print(f"  超过阈值15的数量: {(df['LIVEPERSONNUM'] > 15).sum()}")

threshold = 15
df_cleaned = df.copy()
df_cleaned['LIVEPERSONNUM'] = df_cleaned['LIVEPERSONNUM'].apply(lambda x: min(x, threshold))

print(f"\nLIVEPERSONNUM处理后统计:")
print(f"  最小值: {df_cleaned['LIVEPERSONNUM'].min()}")
print(f"  最大值: {df_cleaned['LIVEPERSONNUM'].max()}")
print(f"  超过阈值15的数量: {(df_cleaned['LIVEPERSONNUM'] > 15).sum()}")

print("\n=== 数据预处理完成 ===")
print(f"清洗后数据行数: {len(df_cleaned)}")

df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
print(f"清洗后数据已保存到: {output_file}")

print("\n清洗后数据前5行预览:")
print(df_cleaned.head())

print("\n清洗后数据描述统计:")
print(df_cleaned.describe())

print("\n清洗后RESULT分布:")
print(df_cleaned['RESULT'].value_counts())