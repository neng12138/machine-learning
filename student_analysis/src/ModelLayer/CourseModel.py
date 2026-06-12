import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AnalysisToolsLayer.CompareModelsPerformance import compare_models_performance
from AnalysisToolsLayer.PermutationFeatureImportance import permutation_feature_importance
from AnalysisToolsLayer.AnalyzeFeatureImportance import analyze_feature_importance
from FairnessLayer.FairnessAnalyzer import FairnessAnalyzer
from FairnessLayer.PlotFairnessAnalysis import plot_fairness_analysis
from FeatureEngineeringLayer.FeatureEngineer import FeatureEngineer
from ModelLayer.OptimizedBaselineModel import OptimizedBaselineModel
from ModelLayer.ComparisonModel import ComparisonModel
from utils.ThresholdAdjustmentTools import compare_threshold_adjustment_performance
from utils.PredictWithGroupThresholds import predict_with_group_thresholds
from utils.GenerateSummary import generate_fairness_summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def train_and_evaluate_course_model(df, course_name, pdf_dirs, report):
    """为单个课程训练和评估模型（集成报告）"""
    
    course_display_name = "数学课程" if course_name == "math" else "葡萄牙语课程"
    print(f"\n🔧 训练{course_display_name}模型...")
    
    # 在报告中添加课程章节
    report.add_header(f"{course_display_name}详细分析", level=3)
    
    # 1. 特征工程
    enhanced_engineer = FeatureEngineer()
    X, y, features = enhanced_engineer.prepare_enhanced_features(df) # 重采样
    X_scaled = enhanced_engineer.scaler.fit_transform(X)
    
    # 2. 模型训练 - 重采样数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 优化基准模型 - 重采样数据集
    optimized_baseline = OptimizedBaselineModel()
    baseline_model = optimized_baseline.advanced_hyperparameter_tuning(X_train, y_train)
    y_pred, y_proba = optimized_baseline.evaluate_with_confidence(X_test, y_test, pdf_dirs[course_name])
    
    # 在报告中记录模型信息
    model_info = f"""
**基准模型配置**:
- 算法: 正则化逻辑回归
- 最佳正则化: {baseline_model.get_params()['penalty'].upper()}
- 正则化强度(C): {baseline_model.get_params()['C']}
- 最佳交叉验证F1: {optimized_baseline.best_score:.4f}
- 测试集F1: {f1_score(y_test, y_pred):.4f}
"""
    report.add_section("模型训练结果", model_info)
    
    # 对比模型 - 重采样数据集
    enhanced_comparison = ComparisonModel()
    comparison_model = enhanced_comparison.train_ensemble_models(X_train, y_train)
    best_model_name = enhanced_comparison.best_model_name  # 获取最佳模型名称
    y_pred_comparison = comparison_model.predict(X_test)
    comparison_f1 = f1_score(y_test, y_pred_comparison)
    print(f"🎯 {course_display_name}最佳对比模型测试集F1分数: {comparison_f1:.4f}")
    
    # 3. 公平性分析
    print(f"\n⚖️ {course_display_name}公平性分析")
    sensitive_attributes = ['sex', 'famsize', 'address', 'Mjob']
    
    # 创建不进行重采样的特征工程实例
    enhanced_engineer_no_resample = FeatureEngineer()
    X_no_resample, y_no_resample, _ = enhanced_engineer_no_resample.prepare_enhanced_features(
        df, handle_imbalance=False   # 不进行重采样
    )
    
    # 使用原始数据分割
    X_train_no_resample, X_test_no_resample, y_train_no_resample, y_test_no_resample, train_indices, test_indices = train_test_split(
        X_no_resample, y_no_resample, df.index, test_size=0.2, random_state=42, stratify=y_no_resample
    )
    # 提取原始测试数据
    test_df_original = df.loc[test_indices]
    
    # 公平性分析预测 - 原始数据集
    X_test_no_resample_scaled = enhanced_engineer.scaler.fit_transform(X_test_no_resample)
    y_pred_fairness = baseline_model.predict(X_test_no_resample_scaled)
    # 提取原始测试数据的预测概率
    y_pred_proba_fairness = baseline_model.predict_proba(X_test_no_resample_scaled)[:, 1]
    
    # 3.1 公平性分析指标计算（原始测试数据）
    fairness_analyzer = FairnessAnalyzer(sensitive_attributes)
    fairness_results_math = fairness_analyzer.calculate_fairness_metrics(
        y_test_no_resample, y_pred_fairness, test_df_original
    )
    fairness_analyzer.generate_fairness_report(fairness_results_math)
    # 修改公平性分析绘图函数调用
    plot_fairness_analysis(fairness_results_math, pdf_dirs[course_name])  # 修改：传递课程特定目录

    # 群体特定阈值调整（传course_name）
    print(f"\n🎯 {course_display_name}群体特定阈值调整")
    y_pred_adjusted, y_pred_proba_adjusted, applied_thresholds = predict_with_group_thresholds(
        baseline_model, X_test_no_resample_scaled, test_df_original, features, course_name, pdf_dirs[course_name]
    )
    
    # 比较调整前后的性能 - 原始数据集
    compare_threshold_adjustment_performance(
        y_test_no_resample, y_pred_fairness, y_pred_adjusted,
        test_df_original, applied_thresholds, pdf_dirs[course_name]  # 修改：传递课程特定目录
    )
    
    # 3.2 公平性分析指标计算（调整阈值后）
    fairness_results = fairness_analyzer.calculate_fairness_metrics(
        y_test_no_resample, y_pred_adjusted, test_df_original
    )
    fairness_analyzer.generate_fairness_report(fairness_results)
    # 修改公平性分析绘图函数调用
    plot_fairness_analysis(fairness_results, pdf_dirs[course_name])  # 修改：传递课程特定目录
    
    # 在报告中记录公平性结果
    fairness_summary = generate_fairness_summary(fairness_results, course_display_name)
    report.add_section("公平性分析结果", fairness_summary)

    # 4. 模型对比分析 - 阈值调整后的基准模型
    print(f"\n📊 {course_display_name}模型对比分析（使用阈值调整后基准模型）")
    # 在相同的测试集上评估所有模型，获取对比模型在公平性测试集上的预测
    # 使用已拟合的scaler进行转换，确保与基准模型使用相同的数据缩放
    X_test_no_resample_scaled_comparison = enhanced_engineer.scaler.transform(X_test_no_resample)
    y_pred_comparison_fairness = comparison_model.predict(X_test_no_resample_scaled_comparison)
    
    # 原始数据集
    comparison_metrics = compare_models_performance(
        y_test_no_resample,  # 使用公平性分析的测试集
        y_pred_adjusted,     # 使用阈值调整后的基准模型预测
        y_pred_comparison_fairness,  # 对比模型在相同测试集上的预测
        model_names=[f'{course_display_name}逻辑回归', f'{course_display_name}{best_model_name}'],
        pdf_dir=pdf_dirs[course_name]
    )

    # 在报告中记录性能对比
    performance_table = [
        ["逻辑回归", f"{comparison_metrics['逻辑回归']['f1_score']:.4f}"],
        [best_model_name, f"{comparison_metrics['集成模型']['f1_score']:.4f}"]
    ]
    report.add_table("模型性能对比", performance_table, ["模型类型", "F1分数"])
    
    # 5. 特征重要性分析
    print(f"\n🔍 {course_display_name}特征重要性分析")
    linear_importance = analyze_feature_importance(
        baseline_model, features, top_n=15, pdf_dir=pdf_dirs[course_name]  # 修改：传递课程特定目录
    )
    
    ensemble_importance = analyze_feature_importance(
        comparison_model, features, top_n=15, pdf_dir=pdf_dirs[course_name]  # 修改：传递课程特定目录
    )
    
    print(f"\n🔍 {course_display_name}排列重要性分析:")
    perm_importance = permutation_feature_importance(
        baseline_model, X_test, y_test, features, pdf_dir=pdf_dirs[course_name]  # 修改：传递课程特定目录
    )

     # 在报告中记录重要特征
    top_features = linear_importance.head(10)[['feature', 'importance']].values.tolist()
    feature_table = [[row[0], f"{row[1]:.4f}"] for row in top_features]
    report.add_table("Top 10重要特征", feature_table, ["特征", "重要性"])
    
    return {
        'course_name': course_name,
        'course_display_name': course_display_name,
        'baseline_model': baseline_model,
        'comparison_model': comparison_model,
        'features': features,
        'fairness_results': fairness_results,
        'performance_comparison': comparison_metrics,
        'feature_importance': {
            'linear': linear_importance,
            'ensemble': ensemble_importance,
            'permutation': perm_importance
        },
        'test_metrics': {
            'f1_baseline_adjusted': f1_score(y_test_no_resample, y_pred_adjusted),
            'f1_baseline_original': f1_score(y_test, y_pred),
            'f1_ensemble': comparison_f1,
            'dataset_size': len(df),
            'pass_rate': df['pass'].mean()
        },
        'threshold_config': applied_thresholds
    }