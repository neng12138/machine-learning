import os
import sys

# 添加文件资源路径，确保能正确找到data目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径
sys.path.append(current_dir)

# 导入主程序的main函数
from concrete_crack_detection import main

# 运行主函数
if __name__ == '__main__':
    main()