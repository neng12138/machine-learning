# 德国信用数据集预处理报告

## 1. 数据集概述

### 1.1 数据来源
本数据集来自德国汉堡大学统计与计量经济学研究所 Hans Hofmann 教授，包含 1000 个信用申请样本。

### 1.2 数据结构

| 属性类型 | 数量 | 特征名称 |
|---------|------|----------|
| 数值型 | 7 | `duration`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `maintenance_people` |
| 分类型 | 13 | `checking_account`, `credit_history`, `purpose`, `savings_account`, `employment`, `personal_status`, `other_debtors`, `property`, `other_installment`, `housing`, `job`, `telephone`, `foreign_worker` |
| 标签 | 1 | `class` (1=好信用, 2=坏信用) |

### 1.3 数据文件
- **原始数据**: `german.csv`
- **数据说明**: `german数据说明.doc`
- **预处理脚本**: `data_preprocessing.py`
- **输出数据**: `german_preprocessed.csv`

---

## 2. 数据预处理步骤

### 2.1 环境准备与数据加载

```python
import pandas as pd
import numpy as np

column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 
    'credit_amount', 'savings_account', 'employment', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installment', 'housing', 'existing_credits', 'job',
    'maintenance_people', 'telephone', 'foreign_worker', 'class'
]

df = pd.read_csv('german.csv', sep=' ', header=None, names=column_names)
```

**数据加载结果**:
- 样本数量: 1000
- 特征数量: 21

---

### 2.2 数据质量检查

#### 2.2.1 缺失值检测

```python
missing_values = df.isnull().sum()
```

**结果**: 无缺失值

#### 2.2.2 重复值检测

```python
duplicate_count = df.duplicated().sum()
```

**结果**: 无重复样本

#### 2.2.3 数据类型检查

| 特征名称 | 数据类型 |
|---------|----------|
| 数值型特征 | int64 |
| 分类型特征 | object |
| 标签 | int64 |

---

### 2.3 数值型特征统计分析

| 特征 | 均值 | 标准差 | 最小值 | 25% | 50% | 75% | 最大值 |
|------|------|--------|--------|-----|-----|-----|--------|
| duration | 20.90 | 12.06 | 4 | 12 | 18 | 24 | 72 |
| credit_amount | 3271.26 | 2822.74 | 250 | 1365.5 | 2319.5 | 3972.25 | 18424 |
| installment_rate | 2.97 | 1.15 | 1 | 2 | 3 | 4 | 4 |
| residence_since | 2.85 | 1.10 | 1 | 2 | 3 | 4 | 4 |
| age | 35.55 | 11.37 | 19 | 27 | 33 | 42 | 75 |
| existing_credits | 1.41 | 0.58 | 1 | 1 | 1 | 2 | 4 |
| maintenance_people | 1.16 | 0.36 | 1 | 1 | 1 | 1 | 2 |

---

### 2.4 异常值检测（IQR方法）

采用四分位距法检测异常值：
- Q1 = 第25百分位数
- Q3 = 第75百分位数
- IQR = Q3 - Q1
- 异常值范围：[Q1 - 1.5×IQR, Q3 + 1.5×IQR]

**检测结果**:

| 特征 | 异常值数量 | 正常值范围 |
|------|----------|------------|
| duration | 20 | [0.00, 42.00] |
| credit_amount | 60 | [-1659.13, 6996.88] |
| installment_rate | 0 | [1.00, 5.00] |
| residence_since | 0 | [0.00, 6.00] |
| age | 10 | [11.50, 57.50] |
| existing_credits | 0 | [0.00, 3.00] |
| maintenance_people | 196 | [0.50, 1.50] |

**处理策略**: 
- `duration`, `credit_amount`, `age`: 异常值在合理业务范围内，保留
- `maintenance_people`: 只有1和2两个取值，196个值为2被误判为异常，保留

---

### 2.5 分类型特征编码

#### 2.5.1 标签解码

将原始编码转换为可读描述：

| 特征 | 编码 | 解码后 |
|------|------|--------|
| checking_account | A11 | <0 DM |
| | A12 | 0-200 DM |
| | A13 | >=200 DM |
| | A14 | no account |
| credit_history | A30 | no credits/all paid |
| | A31 | all paid at this bank |
| | A32 | existing credits paid |
| | A33 | delay in past |
| | A34 | critical account |
| purpose | A40-A410 | car(new), car(used), furniture, radio, appliances, repairs, education, vacation, retraining, business, others |
| savings_account | A61-A65 | <100, 100-500, 500-1000, >=1000, unknown |
| employment | A71-A75 | unemployed, <1y, 1-4y, 4-7y, >=7y |
| personal_status | A91-A95 | 男/离异, 女/离异/已婚, 男/单身, 男/已婚/丧偶, 女/单身 |
| other_debtors | A101-A103 | none, co-applicant, guarantor |
| property | A121-A124 | real estate, building/life, car/other, unknown |
| other_installment | A141-A143 | bank, stores, none |
| housing | A151-A153 | rent, own, for free |
| job | A171-A174 | 失业/非居民, 无技能居民, 技能工人, 管理层/自雇 |
| telephone | A191-A192 | none, yes |
| foreign_worker | A201-A202 | yes, no |

#### 2.5.2 独热编码

```python
df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

**编码结果**: 13个分类型特征展开为45个二进制特征

---

### 2.6 标签处理

```python
df['class'] = df['class'].map({1: 'good', 2: 'bad'})
```

**标签分布**:
- 好信用 (good): 700 (70%)
- 坏信用 (bad): 300 (30%)

---

### 2.7 数据保存

```python
df_dummies.to_csv('german_preprocessed.csv', index=False)
```

---

## 3. 处理前后对比

| 指标 | 处理前 | 处理后 |
|------|--------|--------|
| 样本数量 | 1000 | 1000 |
| 特征数量 | 21 | 53 |
| 数值型特征 | 7 | 7 |
| 分类型特征 | 13 | 45 (独热编码) |
| 缺失值 | 0 | 0 |
| 重复值 | 0 | 0 |

---

## 4. 数据质量问题分析

### 4.1 已解决问题

1. **编码不直观**: 原始数据使用A11、A30等编码，已转换为可读描述
2. **分类特征不适合建模**: 已通过独热编码转换为数值型特征
3. **标签值不直观**: 已将1/2转换为good/bad

### 4.2 潜在问题

1. **类别不平衡**: 好信用占70%，坏信用占30%，建模时需考虑
2. **异常值**: `credit_amount`和`age`存在少量极端值，建议建模前进一步分析
3. **特征分布差异**: 不同特征的数值范围差异较大，建议标准化处理

---

## 5. 输出文件

### 5.1 预处理后数据文件

**文件**: `german_preprocessed.csv`

**格式**: CSV

**内容结构**:
- 第1-7列: 数值型特征 (`duration`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `maintenance_people`)
- 第8-52列: 独热编码特征
- 第53列: 标签 (`class`)

### 5.2 预处理脚本

**文件**: `data_preprocessing.py`

**功能**:
- 数据加载与检查
- 缺失值与重复值检测
- 异常值分析
- 特征编码转换
- 结果保存

---

## 6. 使用说明

### 6.1 运行预处理脚本

```bash
python data_preprocessing.py
```

### 6.2 加载处理后数据

```python
import pandas as pd
df = pd.read_csv('german_preprocessed.csv')
```

---

## 附录：特征详细说明

| 序号 | 特征名称 | 类型 | 说明 |
|------|---------|------|------|
| 1 | checking_account | 分类 | 现有支票账户状态 |
| 2 | duration | 数值 | 信贷期限（月） |
| 3 | credit_history | 分类 | 信贷历史 |
| 4 | purpose | 分类 | 信贷用途 |
| 5 | credit_amount | 数值 | 信贷金额（DM） |
| 6 | savings_account | 分类 | 储蓄账户/债券 |
| 7 | employment | 分类 | 当前就业年限 |
| 8 | installment_rate | 数值 | 分期付款占可支配收入百分比 |
| 9 | personal_status | 分类 | 个人状态与性别 |
| 10 | other_debtors | 分类 | 其他债务人/担保人 |
| 11 | residence_since | 数值 | 当前住所居住年限 |
| 12 | property | 分类 | 财产状况 |
| 13 | age | 数值 | 年龄（岁） |
| 14 | other_installment | 分类 | 其他分期付款计划 |
| 15 | housing | 分类 | 住房状况 |
| 16 | existing_credits | 数值 | 在本银行现有信贷数量 |
| 17 | job | 分类 | 职业 |
| 18 | maintenance_people | 数值 | 需赡养人数 |
| 19 | telephone | 分类 | 是否有电话 |
| 20 | foreign_worker | 分类 | 是否为外籍工人 |
| 21 | class | 标签 | 信用评级（1=好, 2=坏） |