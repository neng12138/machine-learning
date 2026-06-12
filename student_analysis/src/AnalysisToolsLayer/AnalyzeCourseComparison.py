import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def cross_course_comparison(math_results, por_results, pdf_dirs, report):
    """跨课程对比分析（集成报告）"""
    
    print("\n📈 跨课程性能对比:")
    print("=" * 50)
    
    # 性能指标对比
    metrics_comparison = pd.DataFrame({
        '数学课程': [
            math_results['test_metrics']['f1_baseline_adjusted'],
            math_results['test_metrics']['f1_ensemble'],
            math_results['test_metrics']['dataset_size'],
            math_results['test_metrics']['pass_rate']
        ],
        '葡萄牙语课程': [
            por_results['test_metrics']['f1_baseline_adjusted'],
            por_results['test_metrics']['f1_ensemble'],
            por_results['test_metrics']['dataset_size'],
            por_results['test_metrics']['pass_rate']
        ]
    }, index=['逻辑回归F1', '集成模型F1', '数据集大小', '及格率'])
    
    print(metrics_comparison) # 打印对比表格
    
    # 可视化对比
    plt.figure(figsize=(12, 8))
    
    # F1分数对比
    plt.subplot(2, 2, 1)
    courses = ['数学', '葡萄牙语']
    baseline_f1 = [math_results['test_metrics']['f1_baseline_adjusted'], por_results['test_metrics']['f1_baseline_adjusted']]
    ensemble_f1 = [math_results['test_metrics']['f1_ensemble'], por_results['test_metrics']['f1_ensemble']]
    
    x = np.arange(len(courses))
    width = 0.35
    bar1 = plt.bar(x - width/2, baseline_f1, width, label='逻辑回归', alpha=0.7)  # 修改：更新标签
    bar2 = plt.bar(x + width/2, ensemble_f1, width, label='集成模型', alpha=0.7)
    plt.xlabel('课程')
    plt.ylabel('F1分数')
    plt.title('跨课程F1分数对比')
    plt.xticks(x, courses)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 添加数据标签
    for bar in bar1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bar2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 数据集信息对比
    plt.subplot(2, 2, 2)
    sizes = [math_results['test_metrics']['dataset_size'], por_results['test_metrics']['dataset_size']]
    pass_rates = [math_results['test_metrics']['pass_rate'], por_results['test_metrics']['pass_rate']]
    # 为柱状图添加标签
    bars = plt.bar(courses, sizes, alpha=0.7, label='数据集大小', color='lightblue')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # 及格率折线图
    plt.twinx()
    plt.plot(courses, pass_rates, 'o-', color='red', linewidth=2, markersize=8, label='及格率')
    plt.ylabel('及格率')
    plt.title('数据集信息对比')
    plt.legend()
    
    # 公平性对比 - 性别FPR
    plt.subplot(2, 2, 3)
    math_sex_fpr = [math_results['fairness_results']['sex']['F']['fpr'], math_results['fairness_results']['sex']['M']['fpr']]
    por_sex_fpr = [por_results['fairness_results']['sex']['F']['fpr'], por_results['fairness_results']['sex']['M']['fpr']]
    
    x = np.arange(2)
    width = 0.35
    bar_math = plt.bar(x - width/2, math_sex_fpr, width, label='数学课程', alpha=0.7)
    bar_por = plt.bar(x + width/2, por_sex_fpr, width, label='葡萄牙语课程', alpha=0.7)
    plt.xlabel('性别')
    plt.ylabel('假阳性率(FPR)')
    plt.title('跨课程性别FPR对比')
    plt.xticks(x, ['女性', '男性'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 添加数学课程FPR标签
    for bar in bar_math:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=9)
    # 添加葡萄牙语课程FPR标签
    for bar in bar_por:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=9)
    
    # 公平性对比 - 性别FNR
    plt.subplot(2, 2, 4)
    math_sex_fnr = [math_results['fairness_results']['sex']['F']['fnr'], math_results['fairness_results']['sex']['M']['fnr']]
    por_sex_fnr = [por_results['fairness_results']['sex']['F']['fnr'], por_results['fairness_results']['sex']['M']['fnr']]
    
    bar_math_fnr = plt.bar(x - width/2, math_sex_fnr, width, label='数学课程', alpha=0.7)
    bar_por_fnr = plt.bar(x + width/2, por_sex_fnr, width, label='葡萄牙语课程', alpha=0.7)
    plt.xlabel('性别')
    plt.ylabel('假阴性率(FNR)')
    plt.title('跨课程性别FNR对比')
    plt.xticks(x, ['女性', '男性'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 添加数学课程FNR标签
    for bar in bar_math_fnr:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=9)
    # 添加葡萄牙语课程FNR标签
    for bar in bar_por_fnr:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if pdf_dirs:
        save_plot_as_pdf(plt, "cross_course_comparison", pdf_dirs['base'])
    
    plt.show()
    
    # 打印关键发现
    print("\n🔍 关键发现:")
    print(f"- 数学课程数据集: {math_results['test_metrics']['dataset_size']} 条记录")
    print(f"- 葡萄牙语课程数据集: {por_results['test_metrics']['dataset_size']} 条记录")
    print(f"- 数学课程及格率: {math_results['test_metrics']['pass_rate']:.3f}")
    print(f"- 葡萄牙语课程及格率: {por_results['test_metrics']['pass_rate']:.3f}")
    print(f"- 最佳模型F1分数 - 数学: {max(math_results['test_metrics']['f1_baseline_adjusted'], math_results['test_metrics']['f1_ensemble']):.3f}")  # 修改
    print(f"- 最佳模型F1分数 - 葡萄牙语: {max(por_results['test_metrics']['f1_baseline_adjusted'], por_results['test_metrics']['f1_ensemble']):.3f}")  # 修改
    
    # 在报告中添加对比分析
    comparison_data = [
        ["数学课程", f"{math_results['test_metrics']['f1_baseline_adjusted']:.3f}", 
         f"{math_results['test_metrics']['f1_ensemble']:.3f}", 
         f"{math_results['test_metrics']['dataset_size']}"],
        ["葡萄牙语课程", f"{por_results['test_metrics']['f1_baseline_adjusted']:.3f}", 
         f"{por_results['test_metrics']['f1_ensemble']:.3f}", 
         f"{por_results['test_metrics']['dataset_size']}"]
    ]
    
    report.add_table("跨课程性能对比", comparison_data, 
                    ["课程", "逻辑回归F1", "集成模型F1", "数据集大小"])
