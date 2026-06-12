def get_course_specific_thresholds(course_name):
    """获取课程特定的群体阈值配置"""

    if course_name == "math":
        # 数学课程阈值配置
        return {
            'sex_F': 0.4,   # 女性样本的阈值 (减少漏报)
            'sex_M': 0.6,   # 男性样本的阈值 (增加召回率)
            'Fjob_health': 0.55,  # 主作业在health样本的阈值 (增加召回率)
            'Fjob_teacher': 0.55,  # 主作业在teacher样本的阈值 (增加召回率)
            'reason_home': 0.55,  # 主作业在teacher样本的阈值 (增加召回率)
            'guardian_teacher': 0.55,  # 主作业在teacher样本的阈值 (增加召回率)
            'address_U': 0.4,  # Urban样本的阈值 (减少漏报)
        }
    elif course_name == "por":
        # 葡萄牙语课程阈值配置
        return {
            'sex_F': 0.35,  # 女性样本的阈值 (减少漏报)
            'sex_M': 0.35,  # 男性样本的阈值 (减少漏报)
            'address_R': 0.07,  # Router样本的阈值 (减少漏报)
            'Mjob_health': 0.55,  # 主作业在health样本的阈值 (增加召回率)
            'Mjob_teacher': 0.55,  # 主作业在teacher样本的阈值 (增加召回率)
        }
    else:
        # 默认配置（无课程特定阈值）- 0.5
        return {}