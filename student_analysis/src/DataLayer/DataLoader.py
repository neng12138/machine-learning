import pandas as pd

class DataLoader:
    """数据加载器"""
    def __init__(self):
        self.math_df = pd.read_csv("./Data/student/student-mat.csv", delimiter=";")
        self.por_df = pd.read_csv("./Data/student/student-por.csv", delimiter=";")
        
    def explore_data(self):
        """数据探索分析"""
        print("数学数据集形状:", self.math_df.shape)
        print("葡萄牙语数据集形状:", self.por_df.shape)
        print("\n数学数据集基本信息:")
        print(self.math_df.info())
        print("\n缺失值统计:")
        print(self.math_df.isnull().sum())