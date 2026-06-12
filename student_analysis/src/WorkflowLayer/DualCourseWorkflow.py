import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLayer.DataLoader import DataLoader
from DataLayer.TargetVariable import create_target_variable
from ModelLayer.CourseModel import train_and_evaluate_course_model
from AnalysisToolsLayer.AnalyzeCourseComparison import cross_course_comparison
from VisualizationLayer.SetupCoursePdfOutput import setup_course_pdf_output
from utils.ThresholdExplanation import print_threshold_configuration_explanation
from utils.MarkdownReport import report
from utils.GenerateSummary import generate_executive_summary

def dual_course_workflow():
    """双课程独立预测工作流（集成Markdown报告）"""
    
    # 初始化报告
    report.add_header("执行摘要", level=1)
    
    # 设置课程分类PDF输出目录
    pdf_dirs = setup_course_pdf_output()
    
    report.add_section("输出目录结构", f"""
- 数学课程图表: `{pdf_dirs['math']}`
- 葡萄牙语课程图表: `{pdf_dirs['por']}`
- 跨课程对比图表: `{pdf_dirs['base']}`
""")
    
    # 打印阈值配置说明
    print_threshold_configuration_explanation()
    
    # 1. 数据加载
    print("步骤1: 数据加载")
    report.add_header("数据加载与预处理", level=2)
    
    loader = DataLoader()
    # G3二分类目标变量 -- G3>=10: 及格（1）  G3<10: 不及格（0）
    # 创建数学课程目标变量
    math_df = create_target_variable(loader.math_df)
    # 创建葡萄牙语课程目标变量
    por_df = create_target_variable(loader.por_df)
    
    dataset_info = f"""
**数据集统计**:
- 数学课程: {math_df.shape[0]} 条记录, 及格率: {math_df['pass'].mean():.3f}
- 葡萄牙语课程: {por_df.shape[0]} 条记录, 及格率: {por_df['pass'].mean():.3f}
- 总唯一学生数: {len(math_df) + len(por_df) - 382} (382名重叠学生)
"""
    report.add_section("数据集信息", dataset_info)
    
    print(f"数学数据集: {math_df.shape}, 及格率: {math_df['pass'].mean():.3f}")
    print(f"葡萄牙语数据集: {por_df.shape}, 及格率: {por_df['pass'].mean():.3f}")
    
    # 2. 分别训练两个模型
    print("\n" + "="*60)
    print("🎯 数学课程模型训练")
    print("="*60)
    
    report.add_header("数学课程分析", level=2)
    # 训练数学课程模型
    math_results = train_and_evaluate_course_model(
        math_df, "math", pdf_dirs, report
    )
    
    print("\n" + "="*60)
    print("🎯 葡萄牙语课程模型训练") 
    print("="*60)
    
    report.add_header("葡萄牙语课程分析", level=2)
    # 训练葡萄牙语课程模型
    por_results = train_and_evaluate_course_model(
        por_df, "por", pdf_dirs, report
    )
    
    # 3. 跨课程对比分析
    print("\n" + "="*60)
    print("📊 跨课程模型对比分析")
    print("="*60)
    
    report.add_header("跨课程对比分析", level=2)
    # 对比分析数学课程和葡萄牙语课程模型
    cross_course_comparison(math_results, por_results, pdf_dirs, report)
    
    # 4. 生成总结
    generate_executive_summary(math_results, por_results, report)
    
    # 保存报告
    report_filename = report.save_report()
    
    print(f"\n✅ 所有图表已分类保存:")
    print(f"📁 数学课程图表: {pdf_dirs['math']}")
    print(f"📁 葡萄牙语课程图表: {pdf_dirs['por']}")
    print(f"📄 Markdown报告: {report_filename}")
    
    return {
        'math_results': math_results,
        'por_results': por_results,
        'pdf_directories': pdf_dirs,
        'report_filename': report_filename
    }