
def generate_fairness_summary(fairness_results, course_name):
    """生成公平性分析摘要"""
    summary = f"**{course_name}公平性指标**:\n\n"
    
    for attr, metrics in fairness_results.items():
        summary += f"- **{attr}**:\n"
        for group, values in metrics.items():
            summary += f"  - {group}: FPR={values['fpr']:.3f}, FNR={values['fnr']:.3f}, F1={values['f1']:.3f}\n"
        summary += "\n"
    
    return summary


def generate_executive_summary(math_results, por_results, report):
    """生成执行摘要"""
    report.add_header("执行摘要", level=1)
    
    summary = f"""
## 📊 总体性能

**数学课程**:
- 最佳模型F1分数: {max(math_results['test_metrics']['f1_baseline_adjusted'], math_results['test_metrics']['f1_ensemble']):.3f}
- 数据集大小: {math_results['test_metrics']['dataset_size']} 条记录
- 及格率: {math_results['test_metrics']['pass_rate']:.3f}

**葡萄牙语课程**:
- 最佳模型F1分数: {max(por_results['test_metrics']['f1_baseline_adjusted'], por_results['test_metrics']['f1_ensemble']):.3f}
- 数据集大小: {por_results['test_metrics']['dataset_size']} 条记录
- 及格率: {por_results['test_metrics']['pass_rate']:.3f}

## ⚖️ 公平性关键发现

**数学课程阈值配置**:
{format_thresholds(math_results['threshold_config'])}

**葡萄牙语课程阈值配置**:
{format_thresholds(por_results['threshold_config'])}

## 🔍 总结

**数学课程**:
- 基准模型: L1正则化逻辑回归
- 对比模型: SVM
- 数据集大小: {math_results['test_metrics']['dataset_size']} 条记录
- 及格率: {math_results['test_metrics']['pass_rate']:.3f}

**葡萄牙语课程**:
- 基准模型: L1正则化逻辑回归
- 对比模型: Gradient Boosting
- 数据集大小: {por_results['test_metrics']['dataset_size']} 条记录
- 及格率: {por_results['test_metrics']['pass_rate']:.3f}

## 🎯 建议与改进

1. **模型部署**: 改进阈值调整后的逻辑回归模型，平衡性能与公平性
2. **监控机制**: 持续监控各群体的误报率和漏报率
3. **特征优化**: 基于特征重要性分析，进一步优化特征工程
"""

    report.add_section("", summary)


def format_thresholds(thresholds):
    """格式化阈值配置"""
    result = ""
    for group, threshold in thresholds.items():
        result += f"- {group}: {threshold:.2f}\n"
    return result