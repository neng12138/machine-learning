from utils.CourseSpecificThresholds import get_course_specific_thresholds

def print_threshold_configuration_explanation():
    """打印阈值配置说明"""
    print("\n📋 课程特定阈值配置说明:")
    print("=" * 50)
    
    # 数学课程阈值
    math_thresholds = get_course_specific_thresholds("math")
    # 葡萄牙语课程阈值
    por_thresholds = get_course_specific_thresholds("por")
    
    print("数学课程配置:")
    for group, threshold in math_thresholds.items():
        effect = "减少漏报" if "F" in group else "增加召回率"
        print(f"  - {group}: 阈值 {threshold:.2f} ({effect})")
    
    print("\n葡萄牙语课程配置:")
    for group, threshold in por_thresholds.items():
        effect = "进一步减少漏报" if "F" in group else "进一步提高召回率"
        print(f"  - {group}: 阈值 {threshold:.2f} ({effect})")
    
    # print("\n策略说明:")
    # print("  - 女性阈值 < 0.5: 降低标准，识别更多需要帮助的女性学生")
    # print("  - 男性阈值 > 0.5: 提高标准，更准确地识别男性学生风险")
    # print("  - 葡萄牙语课程阈值更激进: 反映不同学科的特点和需求")
    print("=" * 50)