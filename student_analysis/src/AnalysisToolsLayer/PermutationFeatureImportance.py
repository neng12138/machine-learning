import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def permutation_feature_importance(model, X_test, y_test, feature_names, n_repeats=10, pdf_dir=None):
    """使用排列重要性分析特征重要性并保存为PDF"""
    # 计算排列重要性
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats, 
        random_state=42,
        scoring='f1'
    )
    
    # 转换为DataFrame并排序
    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    features_sorted = perm_imp_df['feature'].values
    importance_sorted = perm_imp_df['importance_mean'].values
    std_sorted = perm_imp_df['importance_std'].values
    
    y_pos = np.arange(len(features_sorted))
    plt.barh(y_pos, importance_sorted, xerr=std_sorted, align='center', alpha=0.7, color='lightblue')
    plt.yticks(y_pos, features_sorted)
    plt.xlabel('排列重要性 (F1分数下降)')
    plt.title('排列特征重要性分析')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # 保存为PDF
    if pdf_dir:
        save_plot_as_pdf(plt, "permutation_feature_importance", pdf_dir)
    
    plt.show()
    
    print("\n排列特征重要性分析:")
    print("-" * 50)
    for i, row in perm_imp_df.iterrows():
        print(f"{row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
    
    return perm_imp_df