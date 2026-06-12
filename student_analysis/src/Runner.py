from WorkflowLayer.DualCourseWorkflow import dual_course_workflow

# 运行双课程独立分析
results = dual_course_workflow()

# 访问各个课程的结果
math_model = results['math_results']['baseline_model']
por_model = results['por_results']['baseline_model']

math_fairness = results['math_results']['fairness_results']
por_fairness = results['por_results']['fairness_results']

# 访问PDF目录
math_pdf_dir = results['pdf_directories']['math']
por_pdf_dir = results['pdf_directories']['por']

# 访问报告文件
print(f"报告文件: {results['report_filename']}")
