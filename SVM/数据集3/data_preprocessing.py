import pandas as pd
import numpy as np

column_names = [
    'checking_account',  
    'duration',         
    'credit_history',   
    'purpose',         
    'credit_amount',     
    'savings_account',  
    'employment',       
    'installment_rate', 
    'personal_status',  
    'other_debtors',    
    'residence_since',  
    'property',         
    'age',              
    'other_installment',
    'housing',          
    'existing_credits', 
    'job',              
    'maintenance_people',
    'telephone',        
    'foreign_worker',   
    'class'            
]

numerical_cols = ['duration', 'credit_amount', 'installment_rate', 'residence_since', 'age', 'existing_credits', 'maintenance_people']
categorical_cols = ['checking_account', 'credit_history', 'purpose', 'savings_account', 'employment', 'personal_status', 'other_debtors', 'property', 'other_installment', 'housing', 'job', 'telephone', 'foreign_worker']

df = pd.read_csv('german.csv', sep=' ', header=None, names=column_names)

print("=== 数据集基本信息 ===")
print(f"样本数量: {df.shape[0]}")
print(f"特征数量: {df.shape[1]}")
print("\n=== 数据类型 ===")
print(df.dtypes)

print("\n=== 数值型特征统计描述 ===")
print(df[numerical_cols].describe())

print("\n=== 分类型特征统计 ===")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\n=== 标签分布 ===")
print(df['class'].value_counts())
print(f"好信用占比: {df['class'].value_counts()[1]/len(df):.2%}")
print(f"坏信用占比: {df['class'].value_counts()[2]/len(df):.2%}")

print("\n=== 检查缺失值 ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("无缺失值")

print("\n=== 检查重复值 ===")
duplicate_count = df.duplicated().sum()
print(f"重复样本数量: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"已删除重复样本，剩余样本数: {df.shape[0]}")

print("\n=== 检查异常值 (数值型特征) ===")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: 异常值数量 = {len(outliers)}")
    if len(outliers) > 0:
        print(f"  正常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\n=== 数据编码处理 ===")

df['checking_account'] = df['checking_account'].map({
    'A11': '<0 DM',
    'A12': '0-200 DM',
    'A13': '>=200 DM',
    'A14': 'no account'
})

df['credit_history'] = df['credit_history'].map({
    'A30': 'no credits/all paid',
    'A31': 'all paid at this bank',
    'A32': 'existing credits paid',
    'A33': 'delay in past',
    'A34': 'critical account'
})

df['purpose'] = df['purpose'].map({
    'A40': 'car (new)',
    'A41': 'car (used)',
    'A42': 'furniture/equipment',
    'A43': 'radio/television',
    'A44': 'domestic appliances',
    'A45': 'repairs',
    'A46': 'education',
    'A47': 'vacation',
    'A48': 'retraining',
    'A49': 'business',
    'A410': 'others'
})

df['savings_account'] = df['savings_account'].map({
    'A61': '<100 DM',
    'A62': '100-500 DM',
    'A63': '500-1000 DM',
    'A64': '>=1000 DM',
    'A65': 'unknown/no savings'
})

df['employment'] = df['employment'].map({
    'A71': 'unemployed',
    'A72': '<1 year',
    'A73': '1-4 years',
    'A74': '4-7 years',
    'A75': '>=7 years'
})

df['personal_status'] = df['personal_status'].map({
    'A91': 'male: divorced/separated',
    'A92': 'female: divorced/separated/married',
    'A93': 'male: single',
    'A94': 'male: married/widowed',
    'A95': 'female: single'
})

df['other_debtors'] = df['other_debtors'].map({
    'A101': 'none',
    'A102': 'co-applicant',
    'A103': 'guarantor'
})

df['property'] = df['property'].map({
    'A121': 'real estate',
    'A122': 'building society/life insurance',
    'A123': 'car/other',
    'A124': 'unknown/no property'
})

df['other_installment'] = df['other_installment'].map({
    'A141': 'bank',
    'A142': 'stores',
    'A143': 'none'
})

df['housing'] = df['housing'].map({
    'A151': 'rent',
    'A152': 'own',
    'A153': 'for free'
})

df['job'] = df['job'].map({
    'A171': 'unemployed/unskilled non-resident',
    'A172': 'unskilled resident',
    'A173': 'skilled employee/official',
    'A174': 'management/self-employed/highly qualified'
})

df['telephone'] = df['telephone'].map({
    'A191': 'none',
    'A192': 'yes'
})

df['foreign_worker'] = df['foreign_worker'].map({
    'A201': 'yes',
    'A202': 'no'
})

df['class'] = df['class'].map({1: 'good', 2: 'bad'})

df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_dummies.to_csv('german_preprocessed.csv', index=False)

print("\n=== 预处理完成 ===")
print(f"处理后特征数量: {df_dummies.shape[1]}")
print(f"处理后样本数量: {df_dummies.shape[0]}")
print("\n保存文件: german_preprocessed.csv")