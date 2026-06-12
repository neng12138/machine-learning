import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)

def load_data():
    df1 = pd.read_csv('./数据集1/cleaned_data.csv')
    df2 = pd.read_csv('./数据集2/australian_preprocessed.csv')
    df3 = pd.read_csv('./数据集3/german_preprocessed.csv')
    
    return df1, df2, df3

def plot_class_distribution(df1, df2, df3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    result_counts1 = df1['RESULT'].value_counts()
    axes[0].pie(result_counts1.values, labels=result_counts1.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('数据集1：类别分布')
    
    a15_counts = df2['A15'].value_counts()
    axes[1].pie(a15_counts.values, labels=['1', '0'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('数据集2：类别分布')
    
    class_counts3 = df3['class'].value_counts()
    axes[2].pie(class_counts3.values, labels=class_counts3.index, autopct='%1.1f%%', startangle=90)
    axes[2].set_title('数据集3：类别分布')
    
    plt.tight_layout()
    plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_numeric_distributions(df3):
    numeric_cols = ['duration', 'credit_amount', 'age']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df3[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f'{col} 分布')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('频数')
    
    plt.tight_layout()
    plt.savefig('figures/numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df2):
    plt.figure(figsize=(12, 10))
    corr_matrix = df2.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('数据集2：特征相关性热力图')
    plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_target_relationship(df1):
    features = ['SEX', 'AGE_CLASS', 'MARRYSTATUS', 'EDUCATIONLEVEL', 'LIVE_CLASS', 'WORKYEAR_CLASS', 'INCOME_CLASS']
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        cross_tab = pd.crosstab(df1[feature], df1['RESULT'], normalize='index')
        cross_tab.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'{feature} 与 RESULT 的关系')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('比例')
        axes[i].legend(['Bad', 'Good'])
    
    for j in range(len(features), 9):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/feature_target_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df1, df2, df3 = load_data()
    
    plot_class_distribution(df1, df2, df3)
    print('已生成：类别分布饼图')
    
    plot_numeric_distributions(df3)
    print('已生成：数据集3数值特征分布直方图')
    
    plot_correlation_heatmap(df2)
    print('已生成：数据集2相关性热力图')
    
    plot_feature_target_relationship(df1)
    print('已生成：数据集1特征与目标变量关系图')
    
    print('\n所有可视化图表已保存到 figures/ 目录')

if __name__ == '__main__':
    main()