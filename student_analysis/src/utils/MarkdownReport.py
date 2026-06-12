import datetime

class MarkdownReport:
    """Markdown报告生成器"""

    def __init__(self):
        self.report_content = []
        self.course_results = {}
        self.start_time = datetime.datetime.now()
    
    def add_header(self, title, level=1):
        """添加标题"""
        self.report_content.append(f"{'#' * level} {title}\n")
    
    def add_section(self, title, content, level=2):
        """添加章节"""
        self.report_content.append(f"{'#' * level} {title}\n")
        self.report_content.append(content + "\n")
    
    def add_table(self, title, data, headers):
        """添加表格"""
        self.add_section(title, "", level=3)
        
        # 表头
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["---"] * len(headers)) + "|"
        
        self.report_content.append(header_row)
        self.report_content.append(separator)
        
        # 数据行
        for row in data:
            row_str = "| " + " | ".join(str(x) for x in row) + " |"
            self.report_content.append(row_str)
        
        self.report_content.append("")
    
    def add_code_block(self, code, language="python"):
        """添加代码块"""
        self.report_content.append(f"```{language}")
        self.report_content.append(code)
        self.report_content.append("```\n")
    
    def add_bullet_list(self, items):
        """添加项目列表"""
        for item in items:
            self.report_content.append(f"- {item}")
        self.report_content.append("")
    
    def save_report(self, filename="README.md"):
        """保存报告"""
        # 添加报告头部信息
        header = f"""# 学生学业风险预测分析报告

**生成时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}  
**分析工具**: Python, scikit-learn, pandas, matplotlib, seaborn, numpy  
**数据集**: UCI学生表现数据集（数学 & 葡萄牙语课程）

## 运行指南

**运行方式**: 以压缩文件为根目录，直接运行src目录下的 Runner.py即可

**注意事项**: 
- 确保已安装所有依赖库（根据requirements.txt安装）
- 数据集文件Data应与src目录在同一目录下


"""
        
        full_content = header + "\n".join(self.report_content)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"📄 Markdown报告已保存: {filename}")
        return filename

# 创建全局报告实例
report = MarkdownReport()
