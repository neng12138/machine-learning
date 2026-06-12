import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def compare_models_performance(y_true, y_pred_baseline, y_pred_comparison, model_names=None, pdf_dir=None):
    """对比基准模型和对比模型的性能并保存为PDF"""
    if model_names is None:
        model_names = ['优化逻辑回归', '最佳集成模型']
    
    # 计算各项指标
    metrics = {}
    for i, (name, y_pred) in enumerate(zip(model_names, [y_pred_baseline, y_pred_comparison])):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics[name] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': f1_score(y_true, y_pred),
            # FPR假阳性率 -- 错误将负类识别为正类的比例
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            # FNR假阴性率 -- 错误将正类识别为负类的比例
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'confusion_matrix': cm # 混淆矩阵
        }
        
        # 使用固定键名供CourseModel.py使用
        if i == 0:  # 第一个模型是逻辑回归
            metrics['逻辑回归'] = metrics[name].copy()
        elif i == 1:  # 第二个模型是集成模型
            metrics['集成模型'] = metrics[name].copy()

    # 打印对比结果
    print("=" * 60)
    print("模型性能对比")
    print("=" * 60)
    
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  - 准确率: {model_metrics['accuracy']:.4f}")
        print(f"  - 精确率: {model_metrics['precision']:.4f}")
        print(f"  - 召回率: {model_metrics['recall']:.4f}")
        print(f"  - F1分数: {model_metrics['f1_score']:.4f}")
        print(f"  - 假阳性率: {model_metrics['fpr']:.4f}")
        print(f"  - 假阴性率: {model_metrics['fnr']:.4f}")
    
    # 可视化对比
    plt.figure(figsize=(15, 5))
    
    # F1分数对比
    plt.subplot(1, 3, 1)
    f1_scores = [metrics[name]['f1_score'] for name in model_names]
    bars = plt.bar(model_names, f1_scores, color=['skyblue', 'lightcoral'])
    plt.title('F1分数对比')
    plt.ylabel('F1 Score')
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # 精确率-召回率对比
    plt.subplot(1, 3, 2)
    precision_scores = [metrics[name]['precision'] for name in model_names]
    recall_scores = [metrics[name]['recall'] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    # 绘制柱状图并保存返回值
    precision_bars = plt.bar(x - width/2, precision_scores, width, label='精确率', alpha=0.7)
    recall_bars = plt.bar(x + width/2, recall_scores, width, label='召回率', alpha=0.7)
    plt.title('精确率 vs 召回率')
    plt.xticks(x, model_names)
    plt.legend()
    # 为精确率柱状图添加数值标签
    for bar, score in zip(precision_bars, precision_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    # 为召回率柱状图添加数值标签
    for bar, score in zip(recall_bars, recall_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # 错误率对比
    plt.subplot(1, 3, 3)
    error_rates = [metrics[name]['fpr'] + metrics[name]['fnr'] for name in model_names]
    # 绘制柱状图并保存返回值
    error_bars = plt.bar(model_names, error_rates, color=['lightgreen', 'lightyellow'])
    plt.title('总错误率对比(FPR+FNR)')
    plt.ylabel('错误率')
    # 为错误率柱状图添加数值标签
    for bar, rate in zip(error_bars, error_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存为PDF
    if pdf_dir:
        save_plot_as_pdf(plt, "model_performance_comparison", pdf_dir)
    
    plt.show()
    
    return metrics