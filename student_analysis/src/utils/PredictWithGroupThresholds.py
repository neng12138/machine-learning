import numpy as np
from utils.CourseSpecificThresholds import get_course_specific_thresholds


def predict_with_group_thresholds(model, X_test, test_df_original, features_math, course_name, pdf_dir=None):
    """使用课程特定的群体阈值进行预测"""
    
    # 获取基础预测概率
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 默认阈值 - 0.5
    default_threshold = 0.5
    y_pred_default = (y_pred_proba >= default_threshold).astype(int)
    
    # 获取课程特定的阈值配置
    group_thresholds = get_course_specific_thresholds(course_name)
    
    course_display_name = "数学课程" if course_name == "math" else "葡萄牙语课程"
    print(f"🔍 {course_display_name}群体特定阈值配置:")
    for group_key, threshold in group_thresholds.items():
        print(f"  - {group_key}: {threshold:.2f}")
    
    # 应用群体特定阈值
    y_pred_adjusted = y_pred_default.copy()
    
    for group_key, custom_threshold in group_thresholds.items():
        attr, group_value = group_key.split('_', 1)
        
        if attr in test_df_original.columns:
            group_mask = (test_df_original[attr] == group_value)
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) > 0:
                # 对该群体应用自定义阈值
                group_proba = y_pred_proba[group_indices]
                group_predictions = (group_proba >= custom_threshold).astype(int)
                y_pred_adjusted[group_indices] = group_predictions
                
                print(f"✅ 对 {attr}_{group_value} 应用阈值 {custom_threshold:.2f}")
                print(f"   影响样本数: {len(group_indices)}")
    
    # 返回调整后的预测结果、基础概率和应用的阈值
    return y_pred_adjusted, y_pred_proba, group_thresholds 