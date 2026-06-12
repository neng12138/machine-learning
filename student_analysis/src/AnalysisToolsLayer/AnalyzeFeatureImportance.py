import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def analyze_feature_importance(model, feature_names, top_n=15, pdf_dir=None):
    """分析特征重要性并保存为PDF"""
    
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        importance_type = "系数绝对值"
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_type = "特征重要性"
    else:
        return None
    
    # 创建特征重要性DataFrame
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    feature_imp_top = feature_imp.head(top_n)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_imp_top)))
    
    bars = plt.barh(range(len(feature_imp_top)), 
                   feature_imp_top['importance'], 
                   color=colors)
    plt.yticks(range(len(feature_imp_top)), feature_imp_top['feature'])
    plt.xlabel(f'特征重要性 ({importance_type})')
    plt.title(f'Top {top_n} 特征重要性排名 - {type(model).__name__}')
    plt.gca().invert_yaxis()
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存为PDF
    if pdf_dir:
        model_type = type(model).__name__.lower()
        save_plot_as_pdf(plt, f"feature_importance_{model_type}", pdf_dir)
    
    plt.show()
    
    # 打印特征重要性详情
    print(f"\n特征重要性分析 ({importance_type}):")
    print("-" * 50)
    for i, row in feature_imp_top.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return feature_imp