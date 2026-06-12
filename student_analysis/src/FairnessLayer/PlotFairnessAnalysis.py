import matplotlib.pyplot as plt
import numpy as np
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_fairness_analysis(fairness_results, pdf_dir=None):
    """可视化公平性分析结果并保存为PDF"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    plots_generated = 0
    for i, (attr, metrics) in enumerate(fairness_results.items()):
        if i >= 4:  # 只显示前4个属性
            break
            
        groups = list(metrics.keys())
        fprs = [metrics[g]['fpr'] for g in groups]
        fnrs = [metrics[g]['fnr'] for g in groups]
        
        x = np.arange(len(groups))
        width = 0.35
        
        # 绘制柱状图并保存返回值
        fpr_bars = axes[i].bar(x - width/2, fprs, width, label='假阳性率(FPR)', alpha=0.7, color='red')
        fnr_bars = axes[i].bar(x + width/2, fnrs, width, label='假阴性率(FNR)', alpha=0.7, color='blue')
        
        # 为FPR柱状图添加数值标签
        for bar, fpr in zip(fpr_bars, fprs):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{fpr:.3f}', ha='center', va='bottom')
        # 为FNR柱状图添加数值标签
        for bar, fnr in zip(fnr_bars, fnrs):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{fnr:.3f}', ha='center', va='bottom')

        axes[i].set_xlabel('群体')
        axes[i].set_ylabel('比率')
        axes[i].set_title(f'{attr} - 公平性分析')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(groups)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        plots_generated += 1
    
    # 隐藏多余的子图
    for i in range(plots_generated, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存为PDF
    if pdf_dir:
        save_plot_as_pdf(plt, "fairness_analysis", pdf_dir)
    
    plt.show()