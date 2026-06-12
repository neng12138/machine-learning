# 混凝土裂缝检测手动设计网络层方案

## 运行指南

 1. 确保安装所有本项目的依赖包 （requirements.txt）

 2. 由于数据集文件过于庞大，因此请在运行前先下载数据集文件并将对应文件夹放到data目录下

 数据集链接：https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification

 data目录结构
```
data/
├── Negative
├── Positive
```

 3. 直接运行 Runner.py 即可开始训练和测试

#### 注意：由于电脑性能限制，本项目使用的是cpu

---

## 1. 项目背景

混凝土裂缝检测是结构健康监测的重要组成部分，需要模型能够高效、准确地识别图像中的裂缝。本方案假设项目未使用预训练的MobileNetV2模型，而是采用**手动设计的简单网络层**，针对裂缝检测任务进行优化，适合刚入行的开发者学习和实现。

## 2. 手动设计网络层的设计思路

### 2.1 任务特点分析

混凝土裂缝检测具有以下特点：
- **裂缝特征**：细长的线性结构，需要模型捕捉空间纹理信息
- **图像质量**：数据集图像清晰，噪声较少，适合直接特征提取
- **类别均衡**：正负类比例1:1，无需特殊处理类别不平衡问题

### 2.2 设计原则

基于任务特点和小白友好原则，手动设计网络层遵循以下原则：

| 设计原则 | 具体实现 |
|----------|----------|
| **轻量级设计** | 减少参数量和计算量，适合快速训练和测试 |
| **足够有效** | 保证模型能够准确检测裂缝 |
| **易于调试** | 结构清晰，便于排查问题 |
| **适合学习** | 包含CNN核心组件，便于理解CNN工作原理 |

## 3. 手动设计的网络层具体结构

### 3.1 整体架构

手动设计的网络采用**简单卷积神经网络(CNN)**架构，包含以下主要部分：

1. **输入层**：处理原始图像
2. **特征提取模块**：3层卷积层，提取裂缝特征
3. **分类模块**：全连接层，进行最终分类
4. **辅助层**：ReLU激活函数、池化层等

### 3.2 详细网络结构

| 层名称 | 层类型 | 卷积核大小 | 步幅 | 填充 | 输出通道 | 输出尺寸 | 激活函数 | 说明 |
|--------|--------|------------|------|------|----------|----------|----------|------|
| **输入层** | Input | - | - | - | 3 | 224×224 | - | 输入RGB图像 |
| **卷积层1** | Conv2d | 3×3 | 1 | 1 | 8 | 224×224 | ReLU | 提取基础纹理特征 |
| **最大池化1** | MaxPool2d | 2×2 | 2 | 0 | 8 | 112×112 | - | 降低特征图尺寸 |
| **卷积层2** | Conv2d | 3×3 | 1 | 1 | 16 | 112×112 | ReLU | 提取更多细节特征 |
| **最大池化2** | MaxPool2d | 2×2 | 2 | 0 | 16 | 56×56 | - | 进一步降低尺寸 |
| **卷积层3** | Conv2d | 3×3 | 1 | 1 | 32 | 56×56 | ReLU | 提取高层语义特征 |
| **最大池化3** | MaxPool2d | 2×2 | 2 | 0 | 32 | 28×28 | - | 降低计算量 |
| **全局平均池化** | AvgPool2d | 28×28 | 1 | 0 | 32 | 1×1 | - | 简化特征图 |
| **分类层** | Linear | - | - | - | 2 | 1×1 | Softmax | 输出分类结果 |

### 3.3 完整模型定义

```python
import torch
import torch.nn as nn

class SimpleCrackCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCrackCNN, self).__init__()
        
        # 特征提取层：3个卷积层 + 3个池化层
        self.features = nn.Sequential(
            # 第一层：8通道
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # 输入3通道，输出8通道
            nn.ReLU(inplace=True),  # 激活函数，增加非线性
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层，尺寸减半
            
            # 第二层：16通道
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 输入8通道，输出16通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层：32通道
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输入16通道，输出32通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 全局平均池化：将特征图变为1×1大小
            nn.AvgPool2d(kernel_size=28, stride=1)
        )
        
        # 分类层：直接输出2个类别
        self.classifier = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)  # 提取特征
        x = x.view(x.size(0), -1)  # 展平特征图，变为一维向量
        x = self.classifier(x)  # 分类
        x = self.softmax(x)  # 输出概率
        return x
```

## 4. 手动设计网络层的设计特点

### 4.1 简单易懂的设计

| 设计特点 | 具体实现 | 优势 |
|----------|----------|------|
| **少量卷积层** | 仅3个卷积层 | 结构简单，易于理解 |
| **较少通道数** | 8→16→32 | 参数量少，计算速度快 |
| **无复杂模块** | 去掉残差模块等复杂结构 | 代码简洁，便于调试 |
| **清晰的结构** | 卷积→激活→池化的经典组合 | 适合学习CNN基本原理 |
| **易于实现** | 代码行数少，结构清晰 | 新手容易上手 |

### 4.2 裂缝特征针对性设计

| 设计特点 | 具体实现 | 优势 |
|----------|----------|------|
| **3×3卷积核** | 所有卷积层使用3×3卷积核 | 捕捉裂缝的局部纹理特征 |
| **ReLU激活函数** | 所有卷积层后使用ReLU | 引入非线性，增强特征表达能力 |
| **池化层** | 3个最大池化层 | 降低特征图尺寸，减少计算量 |
| **全局平均池化** | 简化特征图 | 减少参数量，增强泛化能力 |

## 5. 手动设计网络层的参数统计

| 指标 | 数值 | 说明 |
|------|------|------|
| **总参数量** | ~0.1M | 非常少，适合快速训练 |
| **计算复杂度** | ~10 MFLOPs | 计算量小，普通电脑也能快速训练 |
| **模型大小** | ~0.4 MB | 模型体积小，便于保存和分享 |
| **网络深度** | 6层（不包括激活函数） | 深度浅，训练速度快 |
| **最大特征图尺寸** | 224×224 | 输入图像尺寸 |
| **最小特征图尺寸** | 1×1 | 全局平均池化后 |

## 6. 手动设计网络层的训练策略

### 6.1 数据预处理

```python
from torchvision import transforms

# 简单的数据预处理
# 训练集预处理（包含数据增强）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像尺寸为224×224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，增加数据多样性
    transforms.ToTensor(),  # 转换为Tensor格式
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化，简化计算
])

# 测试集预处理（不包含数据增强）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 6.2 优化器选择

| 优化器 | 学习率 | 优势 |
|--------|--------|------|
| **SGD** | 0.01 | 简单易懂，适合学习，训练稳定 |

### 6.3 训练参数

| 参数 | 数值 | 说明 |
|------|------|------|
| **批量大小** | 16 | 适合普通电脑内存，训练稳定 |
| **训练epoch** | 20 | 足够训练，不会花费太多时间 |
| **损失函数** | 交叉熵损失 | 适合二分类任务，简单易用 |

## 7. 模型训练示例代码

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 1. 加载数据集
train_dataset = ImageFolder(root='./dataset/train', transform=train_transform)
test_dataset = ImageFolder(root='./dataset/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 2. 创建模型
model = SimpleCrackCNN(num_classes=2)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD优化器

# 4. 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()  # 模型设为训练模式
    running_loss = 0.0
    
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        
        running_loss += loss.item()
    
    # 打印每个epoch的损失
    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

# 5. 测试模型
model.eval()  # 模型设为评估模式
correct = 0
total = 0

with torch.no_grad():  # 不计算梯度
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

## 8. 手动设计网络层的优势与挑战

### 8.1 优势

| 优势 | 详细说明 |
|------|----------|
| **简单易懂** | 网络结构清晰，适合刚入行的开发者学习 |
| **快速训练** | 参数量少，普通电脑几分钟就能完成训练 |
| **易于调试** | 结构简单，出现问题容易排查 |
| **适合学习** | 包含CNN核心组件，便于理解CNN工作原理 |
| **代码简洁** | 模型定义只有几十行代码，容易实现 |

### 8.2 挑战与应对

| 挑战 | 应对策略 |
|------|----------|
| **特征提取能力有限** | 可以通过增加训练数据、调整学习率来提高性能 |
| **可能出现过拟合** | 可以通过增加数据增强、减少训练epoch来缓解 |
| **准确率可能不是最高** | 作为学习模型，重点在于理解原理，后续可以逐步优化 |

## 9. 性能预期

基于手动设计的简单网络层，预期性能如下：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| **测试准确率** | 95%~98% | 对于简单网络来说，这个准确率已经很好 |
| **训练时间** | 完整数据集约10~20分钟 | 普通电脑就能快速完成 |
| **推理速度** | 约5ms/张（CPU） | 推理速度快，适合测试 |
| **内存占用** | 约200MB | 内存占用低，适合普通电脑 |

## 10. 代码优化建议（进阶学习）

当你熟悉了简单网络后，可以尝试以下优化：

### 10.1 增加网络深度

```python
# 优化建议1：增加一个卷积层
self.features = nn.Sequential(
    # 原有层不变...
    
    # 新增第四层：64通道
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.AvgPool2d(kernel_size=14, stride=1)
)

self.classifier = nn.Linear(64, num_classes)
```

### 10.2 增加数据增强

```python
# 优化建议2：添加更多数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ColorJitter(brightness=0.2),  # 亮度调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

## 11. 结论

手动设计的简单网络层方案非常适合刚入行的开发者学习和实现。这个网络结构简单易懂，参数量少，训练速度快，同时能够达到不错的检测准确率。

通过学习这个简单模型，你可以：
1. 理解CNN的基本工作原理
2. 掌握卷积、激活、池化等核心组件
3. 学会模型训练和测试的基本流程
4. 了解数据预处理和增强的重要性

当你熟悉了这个简单模型后，可以逐步尝试更复杂的网络结构和优化策略，进一步提高模型性能。
