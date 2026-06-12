import os

def save_plot_as_pdf(plt, filename, pdf_dir, dpi=300, bbox_inches='tight'):
    """通用函数：保存图表为PDF"""
    filepath = os.path.join(pdf_dir, f"{filename}.pdf")
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format='pdf')
    print(f"📄 图表已保存为: {filepath}")
    return filepath
