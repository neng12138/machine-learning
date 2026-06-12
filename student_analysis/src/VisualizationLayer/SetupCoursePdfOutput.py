import os

def setup_course_pdf_output():
    """设置课程分类的PDF输出目录"""
    base_dir = "Output"
    math_dir = os.path.join(base_dir, "math")
    por_dir = os.path.join(base_dir, "pro")
    
    # 创建目录结构
    directories = [base_dir, math_dir, por_dir]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
    
    # 清空目录中的旧文件（可选）
    for course_dir in [math_dir, por_dir]:
        for file in os.listdir(course_dir):
            if file.endswith('.pdf'):
                os.remove(os.path.join(course_dir, file))
    
    # 返回目录映射
    return {
        'math': math_dir,
        'por': por_dir,
        'base': base_dir
    }