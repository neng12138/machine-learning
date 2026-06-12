def create_target_variable(df, threshold=10):
    """
    根据G3成绩创建二分类目标变量
    G3 >= 10: 及格(1), G3 < 10: 不及格(0)
    """
    df['pass'] = (df['G3'] >= threshold).astype(int)
    return df