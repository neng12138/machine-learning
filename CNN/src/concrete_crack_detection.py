import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 1. 数据加载和预处理
class ConcreteCrackDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, use_canny=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_canny = use_canny
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_canny:
            # 使用Canny边缘检测
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            # 将单通道边缘图转换为三通道
            image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和标签
        return image, label


# 2. 数据预处理和增强
def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # 图片转为 224x224
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 返回训练集和测试集的转换
    return train_transform, test_transform


# 3. 模型定义
class LightweightCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LightweightCNN, self).__init__()
        # 使用MobileNetV2作为基础模型，使用推荐的weights参数代替pretrained
        self.base_model = mobilenet_v2(weights="IMAGENET1K_V1")
        # 替换最后一层分类器
        in_features = self.base_model.classifier[1].in_features
        # 替换为新的分类层
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # 前向传播
        return self.base_model(x)


# 4. 训练函数（添加学习率调度器和早停机制）
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_path = 'best_model.pth'
    
    # 添加学习率调度器 - 学习率衰减策略，当验证损失不再下降时，将学习率衰减为原来的0.1倍
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # 添加早停机制 - 当验证损失连续3个epoch未下降时，停止训练
    patience = 3
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练循环
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 验证循环
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # 打印训练和验证结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        # 更新学习率 - 根据验证损失调整学习率
        scheduler.step(val_epoch_loss)
        
        # 保存最佳模型 - 仅保存验证准确率最高的模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with Val Acc: {best_val_acc:.4f}')
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            print(f'Early stop counter: {early_stop_counter}/{patience}')
        
        # 检查早停条件 - 如果验证损失连续3个epoch未下降，停止训练
        if early_stop_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }


# 5. 评估函数（添加ROC曲线和PR曲线支持）
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    # 测试循环
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)
    
    # 生成分类报告
    print('Classification Report:')
    print(classification_report(all_labels, all_preds))
    
    return test_acc, cm, all_labels, all_preds, all_probs


# 6. 绘制结果曲线
def plot_results(results, title='Model Training Results'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制Loss曲线
    ax1.plot(results['train_losses'], label='Train Loss')
    ax1.plot(results['val_losses'], label='Val Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Accuracy曲线
    ax2.plot(results['train_accs'], label='Train Acc')
    ax2.plot(results['val_accs'], label='Val Acc')
    ax2.set_title('Accuracy vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()


# 7. 绘制混淆矩阵
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()


# 8. 绘制ROC曲线和PR曲线
def plot_roc_pr_curve(labels, probs, title='ROC and PR Curves'):
    # 提取正类概率
    y_true = np.array(labels)
    y_probs = np.array(probs)[:, 1]  # 正类的概率
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    # 绘制ROC曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    
    # 绘制PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {title}')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()
    
    return roc_auc, avg_precision


# 辅助函数：准备Logistic Regression数据 （可选是否使用Canny算子）
def prepare_lr_data(image_paths, labels, use_canny=False):
    images = []
    # 遍历所有图像路径
    for img_path in tqdm(image_paths, desc='Preparing LR data'):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        if use_canny:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        images.append(img)
    # 返回处理后的图像数组和标签数组
    return np.array(images), np.array(labels)


# 9. 5折交叉验证
def k_fold_cross_validation(image_paths, labels, k=5, use_canny=False):
    from sklearn.model_selection import KFold
    
    # 确保image_paths和labels长度一致 （5折交叉验证要求样本数能被k整除）
    assert len(image_paths) == len(labels), f'image_paths和labels长度不一致：{len(image_paths)} vs {len(labels)}'
    
    # 直接使用数据长度创建KFold分割器，而不是索引列表
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    lr_accs = []    # 存储每个折叠的Logistic Regression准确率
    cnn_accs = []     # 存储每个折叠的CNN准确率
    
    # 遍历每个折叠
    for fold, (train_idx, test_idx) in enumerate(kf.split(image_paths)):
        print(f'\n=== Fold {fold+1}/{k} ===')
        
        # 划分训练集和测试集
        fold_train_images = [image_paths[i] for i in train_idx]
        fold_train_labels = [labels[i] for i in train_idx]
        fold_test_images = [image_paths[i] for i in test_idx]
        fold_test_labels = [labels[i] for i in test_idx]
        
        # 获取数据增强和转换
        train_transform, test_transform = get_transforms()
        
        lr_train_imgs, lr_train_lbls = prepare_lr_data(fold_train_images, fold_train_labels, use_canny=use_canny)
        lr_test_imgs, lr_test_lbls = prepare_lr_data(fold_test_images, fold_test_labels, use_canny=use_canny)
        
        # 训练Logistic Regression模型 （逻辑回归模型）
        lr_model, lr_test_acc = train_logistic_regression(lr_train_imgs, lr_train_lbls, lr_test_imgs, lr_test_lbls)
        lr_accs.append(lr_test_acc)
        
        # 创建CNN数据集和数据加载器 （卷积神经网络模型）
        fold_train_dataset = ConcreteCrackDataset(fold_train_images, fold_train_labels, transform=train_transform, use_canny=use_canny)
        fold_test_dataset = ConcreteCrackDataset(fold_test_images, fold_test_labels, transform=test_transform, use_canny=use_canny)
        
        batch_size = 32
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        fold_test_loader = DataLoader(fold_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 训练CNN模型
        cnn_model = LightweightCNN(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 简化训练（5个epoch）
        cnn_model.to(device)
        for epoch in range(5):
            cnn_model.train()
            for images, batch_labels in fold_train_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = cnn_model(images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        # 评估CNN模型
        test_acc, cm, _, _, _ = evaluate_model(cnn_model, fold_test_loader, device=device)
        cnn_accs.append(test_acc)
    
    # 打印5折交叉验证结果
    print(f'\n=== 5-Fold Cross Validation Results ===')
    print(f'Logistic Regression - Mean Accuracy: {np.mean(lr_accs):.4f}, Std: {np.std(lr_accs):.4f}')
    print(f'CNN - Mean Accuracy: {np.mean(cnn_accs):.4f}, Std: {np.std(cnn_accs):.4f}')
    
    # 返回5折交叉验证结果
    return lr_accs, cnn_accs


# 10. Logistic Regression Baseline （逻辑回归模型）
def train_logistic_regression(train_images, train_labels, test_images, test_labels):
    # 减小图像尺寸以减少内存使用
    train_images_small = []
    for img in train_images:
        img_small = cv2.resize(img, (64, 64))
        train_images_small.append(img_small)
    train_images_small = np.array(train_images_small)
    
    test_images_small = []
    for img in test_images:
        img_small = cv2.resize(img, (64, 64))
        test_images_small.append(img_small)
    test_images_small = np.array(test_images_small)
    
    # 将图像展平
    train_images_flat = train_images_small.reshape(train_images_small.shape[0], -1)
    test_images_flat = test_images_small.reshape(test_images_small.shape[0], -1)
    
    # 标准化数据
    scaler = StandardScaler()
    train_images_flat = scaler.fit_transform(train_images_flat)
    test_images_flat = scaler.transform(test_images_flat)
    
    # 训练Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_images_flat, train_labels)
    
    # 评估模型
    train_acc = lr.score(train_images_flat, train_labels)
    test_acc = lr.score(test_images_flat, test_labels)
    
    print(f'Logistic Regression - Train Accuracy: {train_acc:.4f}')
    print(f'Logistic Regression - Test Accuracy: {test_acc:.4f}')
    
    # 生成混淆矩阵
    test_preds = lr.predict(test_images_flat)
    cm = confusion_matrix(test_labels, test_preds)
    print('Confusion Matrix:')
    print(cm)
    
    # 生成分类报告
    print('Classification Report:')
    print(classification_report(test_labels, test_preds))
    
    return lr, test_acc


# 11. 主函数
def main():
    # 获取所有图像路径
    positive_images = glob.glob('data/Positive/*.jpg')
    negative_images = glob.glob('data/Negative/*.jpg')
    
    # 创建标签
    positive_labels = [1] * len(positive_images)
    negative_labels = [0] * len(negative_images)
    
    # 使用数据集子集进行测试（从正类和负类中分别取5%的数据）
    positive_subset_size = int(len(positive_images) * 0.05)
    negative_subset_size = int(len(negative_images) * 0.05)
    
    # 从正类和负类中分别采样
    subset_positive_images = positive_images[:positive_subset_size]
    subset_negative_images = negative_images[:negative_subset_size]
    
    # 合并数据
    all_images = subset_positive_images + subset_negative_images
    all_labels = [1] * len(subset_positive_images) + [0] * len(subset_negative_images)
    
    # 数据集划分
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.25, random_state=42, stratify=train_labels
    )
    
    # 打印数据集大小
    print(f'Train size: {len(train_images)}')       # 训练集大小
    print(f'Validation size: {len(val_images)}')        # 验证集大小
    print(f'Test size: {len(test_images)}')     # 测试集大小
    
    # 获取数据增强和转换
    train_transform, test_transform = get_transforms()
    
    # 添加Canny边缘检测效果可视化
    def visualize_canny_effect(image_paths, num_images=5):
        plt.figure(figsize=(15, 6))
        for i in range(num_images):
            # 读取原始图像
            img_path = image_paths[i]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # 生成Canny边缘图像
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # 绘制原始图像
            plt.subplot(2, num_images, i+1)
            plt.imshow(img)
            plt.title(f'Original {i+1}')
            plt.axis('off')
            
            # 绘制Canny边缘图像
            plt.subplot(2, num_images, i+1+num_images)
            plt.imshow(edges_rgb)
            plt.title(f'Canny {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('canny_vs_original.png')
        plt.show()
    
    # 可视化Canny边缘检测效果
    visualize_canny_effect(all_images[:10], num_images=5)
    
    # 存储不同输入类型的结果
    results_dict = {}
    
    # 同时使用原始RGB图像和Canny边缘图像进行训练和测试
    for use_canny in [False, True]:
        print(f'\n========================================')
        print(f'=== Training with Canny={use_canny} ===')
        print(f'========================================')
        
        # 创建数据集和数据加载器
        train_dataset = ConcreteCrackDataset(train_images, train_labels, transform=train_transform, use_canny=use_canny)
        val_dataset = ConcreteCrackDataset(val_images, val_labels, transform=test_transform, use_canny=use_canny)
        test_dataset = ConcreteCrackDataset(test_images, test_labels, transform=test_transform, use_canny=use_canny)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 训练Logistic Regression Baseline
        print('\n=== Training Logistic Regression Baseline ===')
        
        lr_train_imgs, lr_train_lbls = prepare_lr_data(train_images, train_labels, use_canny=use_canny)
        lr_test_imgs, lr_test_lbls = prepare_lr_data(test_images, test_labels, use_canny=use_canny)
        
        lr_model, lr_test_acc = train_logistic_regression(lr_train_imgs, lr_train_lbls, lr_test_imgs, lr_test_lbls)
        
        # 训练深度模型
        print('\n=== Training Lightweight CNN Model ===')
        model = LightweightCNN(num_classes=2)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 检查设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        
        # 训练模型
        results = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_model.pth'))
        
        # 评估模型
        print('\n=== Evaluating Best Model ===')
        test_acc, cm, eval_labels, eval_preds, eval_probs = evaluate_model(model, test_loader, device=device)
        
        # 绘制结果曲线 - CNN模型
        plot_results(results, title=f'CNN Model Results (Canny={use_canny})')

        # 保存结果
        results_dict[use_canny] = {
            'lr_test_acc': lr_test_acc,
            'cnn_test_acc': test_acc,
            'results': results
        }
    
    # 对比不同输入类型的结果
    print('\n========================================')
    print('=== Comparison of Input Types ===')
    print('========================================')
    print(f'Logistic Regression - Original RGB: {results_dict[False]["lr_test_acc"]:.4f}')
    print(f'Logistic Regression - Canny Edges: {results_dict[True]["lr_test_acc"]:.4f}')
    print(f'CNN - Original RGB: {results_dict[False]["cnn_test_acc"]:.4f}')
    print(f'CNN - Canny Edges: {results_dict[True]["cnn_test_acc"]:.4f}')
    

    # 运行5折交叉验证（使用原始RGB图像）
    print('\n========================================')
    print('=== 5-Fold Cross Validation (Original RGB) ===')
    print('========================================')
    # 确保image_paths和labels长度一致
    assert len(all_images) == len(all_labels), f'image_paths和labels长度不一致：{len(all_images)} vs {len(all_labels)}'
    lr_cv_accs, cnn_cv_accs = k_fold_cross_validation(all_images, all_labels, k=5, use_canny=False)
    
    # 运行5折交叉验证（使用Canny边缘图像）
    print('\n========================================')
    print('=== 5-Fold Cross Validation (Canny Edges) ===')
    print('========================================')
    lr_cv_accs_canny, cnn_cv_accs_canny = k_fold_cross_validation(all_images, all_labels, k=5, use_canny=True)
    
    # 超参数调优（学习率和批量大小）
    print('\n========================================')
    print('=== Hyperparameter Tuning ===')
    print('========================================')
    
    # 定义超参数搜索空间
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [16, 32, 64]
    
    best_lr = None
    best_batch_size = None
    best_acc = 0.0
    
    # 遍历超参数组合
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f'\n--- Testing LR={lr}, Batch Size={batch_size} ---')
            
            # 使用原始RGB图像
            use_canny = False
            
            # 创建数据集和数据加载器
            train_dataset = ConcreteCrackDataset(train_images, train_labels, transform=train_transform, use_canny=use_canny)
            val_dataset = ConcreteCrackDataset(val_images, val_labels, transform=test_transform, use_canny=use_canny)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # 训练模型（CNN模型）
            model = LightweightCNN(num_classes=2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 简化训练（3个epoch）
            model.to(device)
            for epoch in range(3):
                model.train()
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # 验证模型
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            print(f'Validation Accuracy: {val_acc:.4f}')
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr
                best_batch_size = batch_size
    
    # 打印最佳超参数
    print(f'\nBest Hyperparameters: LR={best_lr}, Batch Size={best_batch_size}')
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    
    # 使用最佳超参数重新训练模型并进行详细评估
    print('\n========================================')
    print('=== Final Model Evaluation with Best Hyperparameters ===')
    print('========================================')
    
    # 重新训练模型
    use_canny = False
    train_dataset = ConcreteCrackDataset(train_images, train_labels, transform=train_transform, use_canny=use_canny)
    val_dataset = ConcreteCrackDataset(val_images, val_labels, transform=test_transform, use_canny=use_canny)
    test_dataset = ConcreteCrackDataset(test_images, test_labels, transform=test_transform, use_canny=use_canny)
    
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, num_workers=4)
    
    # 初始化CNN模型
    model = LightweightCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # 训练模型
    print('\n--- Training Final Model ---')
    results = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 详细评估模型
    print('\n--- Detailed Model Evaluation ---')
    test_acc, cm, final_labels, final_preds, final_probs = evaluate_model(model, test_loader, device=device)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, title='Final Model Confusion Matrix')
    
    # 绘制ROC曲线和PR曲线
    roc_auc, avg_precision = plot_roc_pr_curve(final_labels, final_probs, title='Final Model')
    
    # 生成最终结果汇总
    print('\n========================================')
    print('=== Final Results Summary ===')
    print('========================================')
    print(f'Best Hyperparameters: LR={best_lr}, Batch Size={best_batch_size}')
    print(f'Logistic Regression - Test Accuracy: {results_dict[False]["lr_test_acc"]:.4f}')
    print(f'CNN - Test Accuracy: {test_acc:.4f}')
    print(f'CNN - ROC AUC: {roc_auc:.4f}')
    print(f'CNN - Average Precision: {avg_precision:.4f}')
    print(f'5-Fold CV CNN (Original RGB) - Mean Accuracy: {np.mean(cnn_cv_accs):.4f}, Std: {np.std(cnn_cv_accs):.4f}')
    print(f'5-Fold CV CNN (Canny Edges) - Mean Accuracy: {np.mean(cnn_cv_accs_canny):.4f}, Std: {np.std(cnn_cv_accs_canny):.4f}')

