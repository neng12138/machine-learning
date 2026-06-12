import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path):
    column_names = [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 
        'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'
    ]
    df = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
    return df

def analyze_data(df):
    print("=" * 60)
    print("数据集基本信息")
    print("=" * 60)
    print(f"数据集大小: {df.shape[0]} 行, {df.shape[1]} 列")
    print("\n数据类型:")
    print(df.dtypes)
    print("\n前5行数据:")
    print(df.head())
    print("\n统计摘要:")
    print(df.describe())
    print("\n类别属性分布:")
    categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12', 'A15']
    for col in categorical_cols:
        print(f"\n{col} 分布:")
        print(df[col].value_counts())

def check_missing_values(df):
    print("\n" + "=" * 60)
    print("缺失值检查")
    print("=" * 60)
    missing_mask = (df == '?') | (df == 'NA') | (df.isna())
    missing_count = missing_mask.sum()
    print("缺失值数量:")
    print(missing_count[missing_count > 0])
    return missing_mask

def check_duplicates(df):
    print("\n" + "=" * 60)
    print("重复值检查")
    print("=" * 60)
    duplicate_count = df.duplicated().sum()
    print(f"重复行数量: {duplicate_count}")
    if duplicate_count > 0:
        print("\n重复行索引:")
        print(df[df.duplicated()].index.tolist())
    return df.duplicated()

def check_outliers(df):
    print("\n" + "=" * 60)
    print("异常值检查 (Z-score > 3)")
    print("=" * 60)
    numeric_cols = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    outliers = pd.DataFrame()
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_mask = z_scores > 3
        outliers[col] = outlier_mask
        outlier_count = outlier_mask.sum()
        print(f"{col}: {outlier_count} 个异常值 ({(outlier_count/len(df))*100:.2f}%)")
    
    return outliers

def handle_missing_values(df):
    print("\n" + "=" * 60)
    print("处理缺失值")
    print("=" * 60)
    
    categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    numeric_cols = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].replace(['?', 'NA'], mode_val)
        print(f"{col}: 用众数 {mode_val} 填充缺失值")
    
    for col in numeric_cols:
        mean_val = df[col].mean()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(mean_val)
        print(f"{col}: 用均值 {mean_val:.2f} 填充缺失值")
    
    return df

def handle_duplicates(df):
    print("\n" + "=" * 60)
    print("处理重复值")
    print("=" * 60)
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)
    print(f"移除了 {removed_count} 个重复行")
    return df

def handle_outliers(df, method='iqr'):
    print("\n" + "=" * 60)
    print(f"处理异常值 ({method} 方法)")
    print("=" * 60)
    numeric_cols = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            df.loc[outlier_mask, col] = np.where(
                df.loc[outlier_mask, col] < lower_bound, 
                lower_bound, 
                upper_bound
            )
            print(f"{col}: 处理了 {outlier_count} 个异常值")
    
    return df

def normalize_data(df):
    print("\n" + "=" * 60)
    print("数据标准化 (Z-score)")
    print("=" * 60)
    numeric_cols = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        print(f"{col}: 已标准化 (均值={mean:.2f}, 标准差={std:.2f})")
    
    return df

def main():
    input_file = 'australian.csv'
    output_file = 'australian_preprocessed.csv'
    
    print("加载数据...")
    df = load_data(input_file)
    
    analyze_data(df)
    
    check_missing_values(df)
    
    check_duplicates(df)
    
    check_outliers(df)
    
    print("\n" + "=" * 60)
    print("开始数据预处理")
    print("=" * 60)
    
    df_cleaned = handle_missing_values(df.copy())
    
    df_cleaned = handle_duplicates(df_cleaned)
    
    df_cleaned = handle_outliers(df_cleaned)
    
    df_normalized = normalize_data(df_cleaned.copy())
    
    df_normalized.to_csv(output_file, index=False)
    print(f"\n预处理完成！结果已保存到 {output_file}")
    
    print("\n" + "=" * 60)
    print("预处理后数据摘要")
    print("=" * 60)
    print(f"数据集大小: {df_normalized.shape[0]} 行, {df_normalized.shape[1]} 列")
    print("\n统计摘要:")
    print(df_normalized.describe())

if __name__ == "__main__":
    main()