import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def compare_threshold_adjustment_performance(y_test_no_resample, y_pred_math_fairness, y_pred_adjusted, 
                                           test_df_math_original, applied_thresholds, pdf_dir=None):
    """比较阈值调整前后的性能"""
    
    print("\n📊 阈值调整前后性能对比:")
    print("=" * 50)

    # 调整前性能（混淆矩阵）
    cm_before = confusion_matrix(y_test_no_resample, y_pred_math_fairness)
    tn_before, fp_before, fn_before, tp_before = cm_before.ravel()
    fpr_before = fp_before / (fp_before + tn_before) if (fp_before + tn_before) > 0 else 0
    fnr_before = fn_before / (fn_before + tp_before) if (fn_before + tp_before) > 0 else 0

    # 调整后性能（混淆矩阵）
    cm_after = confusion_matrix(y_test_no_resample, y_pred_adjusted)
    tn_after, fp_after, fn_after, tp_after = cm_after.ravel()
    fpr_after = fp_after / (fp_after + tn_after) if (fp_after + tn_after) > 0 else 0
    fnr_after = fn_after / (fn_after + tp_after) if (fn_after + tp_after) > 0 else 0

    # 输出调整前后信息
    print(f"调整前 - FPR: {fpr_before:.3f}, FNR: {fnr_before:.3f}, F1: {f1_score(y_test_no_resample, y_pred_math_fairness):.3f}")
    print(f"调整后 - FPR: {fpr_after:.3f}, FNR: {fnr_after:.3f}, F1: {f1_score(y_test_no_resample, y_pred_adjusted):.3f}")

    # 特定群体性能变化分析
    print("\n🎯 特定群体性能变化:")
    for group_key, threshold in applied_thresholds.items():
        attr, group_value = group_key.split('_', 1)
        group_mask = (test_df_math_original[attr] == group_value)
        
        if group_mask.sum() > 0:
            group_y_true = y_test_no_resample[group_mask]
            group_pred_before = y_pred_math_fairness[group_mask]
            group_pred_after = y_pred_adjusted[group_mask]
            
            # 计算性能指标
            fpr_before_group = calculate_fpr(group_y_true, group_pred_before)
            fnr_before_group = calculate_fnr(group_y_true, group_pred_before)
            fpr_after_group = calculate_fpr(group_y_true, group_pred_after)
            fnr_after_group = calculate_fnr(group_y_true, group_pred_after)
            
            print(f"{group_key}:")
            print(f"  调整前 - FPR: {fpr_before_group:.3f}, FNR: {fnr_before_group:.3f}")
            print(f"  调整后 - FPR: {fpr_after_group:.3f}, FNR: {fnr_after_group:.3f}")
            print(f"  阈值: {threshold:.2f}")

    # 保存调整前后的对比
    plot_threshold_comparison(y_test_no_resample, y_pred_math_fairness, y_pred_adjusted, pdf_dir)


def calculate_fpr(y_true, y_pred):
    """计算假阳性率 - FPR"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def calculate_fnr(y_true, y_pred):
    """计算假阴性率 - FNR"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0


def plot_threshold_comparison(y_test, y_pred_before, y_pred_after, pdf_dir=None):
    """绘制阈值调整前后的对比图"""
    plt.figure(figsize=(12, 6))

    # 调整前
    plt.subplot(1, 2, 1)
    cm_before = confusion_matrix(y_test, y_pred_before)
    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues')
    plt.title('调整前 - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    # 修改混淆矩阵的x轴和y轴标签，将0改为不及格，1改为及格
    plt.xticks([0.5, 1.5], ['不及格', '及格'])
    plt.yticks([0.5, 1.5], ['不及格', '及格']) 

    # 调整后
    plt.subplot(1, 2, 2)
    cm_after = confusion_matrix(y_test, y_pred_after)
    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Blues')
    plt.title('调整后 - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    # 修改混淆矩阵的x轴和y轴标签，将0改为不及格，1改为及格
    plt.xticks([0.5, 1.5], ['不及格', '及格'])
    plt.yticks([0.5, 1.5], ['不及格', '及格']) 

    plt.tight_layout()
    if pdf_dir:
        save_plot_as_pdf(plt, "threshold_adjustment_comparison", pdf_dir)
    plt.show()