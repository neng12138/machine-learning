import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score , roc_auc_score
from VisualizationLayer.SavePlotAsPdf import save_plot_as_pdf
from sklearn.metrics import precision_recall_curve, roc_curve, auc

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class OptimizedBaselineModel:
    """优化基准模型，用于超参数调优和模型评估。"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = None
    
    def advanced_hyperparameter_tuning(self, X_train, y_train):
        """超参数调优"""

        # 扩展参数网格
        param_grid = [
            {
                # L1正则化 - 绝对值
                'penalty': ['l1'],
                'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced', {0: 2, 1: 1}, {0: 1, 1: 2}]
            },
            {
                # L2正则化 - 平方
                'penalty': ['l2'],
                'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
                'solver': ['liblinear', 'lbfgs', 'newton-cg'],
                'class_weight': [None, 'balanced', {0: 2, 1: 1}, {0: 1, 1: 2}]
            },
            {
                # ElasticNet正则化（结合L1和L2）- 绝对值+平方
                'penalty': ['elasticnet'],
                'C': [0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'solver': ['saga'],
                'class_weight': [None, 'balanced']
            }
        ]
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=2000),
            param_grid,
            cv=5,
            scoring='f1', # 评估目标：F1分数
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        
        # 训练模型
        grid_search.fit(X_train, y_train)
        
        # 保存最佳模型、参数和分数
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_  # 保存最佳分数
        
        # 网格搜索后，获得最佳正则化方式和求解器
        best_penalty = self.best_params['penalty']
        best_solver = self.best_params['solver']
        print(f"🎯 优化后的最佳正则化方式: {best_penalty}")
        print(f"🎯 优化后的最佳求解器: {best_solver}")

        print("🎯 优化后的最佳参数:", self.best_params)
        print("📊 最佳交叉验证F1分数:", grid_search.best_score_)
        
        # 显示交叉验证结果
        cv_results = grid_search.cv_results_
        best_index = grid_search.best_index_
        print(f"📈 最佳参数在{len(cv_results['mean_test_score'])}种组合中排名第{best_index + 1}")
        
        return self.model # 返回最佳模型
    
    def evaluate_with_confidence(self, X_test, y_test, pdf_dir=None):
        """模型评估"""
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算多种指标
        f1 = f1_score(y_test, y_pred)       # F1分数
        accuracy = accuracy_score(y_test, y_pred)   # 准确率
        precision = precision_score(y_test, y_pred) # 精确率
        recall = recall_score(y_test, y_pred)   # 召回率
        roc_auc = roc_auc_score(y_test, y_pred_proba)   # ROC AUC
        
        print("🎯 详细性能指标:")
        print(f"  - F1分数: {f1:.4f}")
        print(f"  - 准确率: {accuracy:.4f}")
        print(f"  - 精确率: {precision:.4f}")
        print(f"  - 召回率: {recall:.4f}")
        print(f"  - ROC AUC: {roc_auc:.4f}")
        
        # 绘制综合评估图
        self.plot_comprehensive_evaluation(y_test, y_pred, y_pred_proba, pdf_dir)
        
        return y_pred, y_pred_proba
    
    def plot_comprehensive_evaluation(self, y_test, y_pred, y_pred_proba, pdf_dir):
        """绘制综合评估图表"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('混淆矩阵-逻辑回归')
        # 修改混淆矩阵的x轴和y轴标签，将0改为不及格，1改为及格
        axes[0, 0].set_xlabel('预测类别')
        axes[0, 0].set_ylabel('真实类别')
        axes[0, 0].set_xticklabels(['不及格', '及格'])
        axes[0, 0].set_yticklabels(['不及格', '及格']) 
        
        # 2. ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('假正率')
        axes[0, 1].set_ylabel('真正率')
        axes[0, 1].set_title('ROC曲线')
        axes[0, 1].legend(loc="lower right")
        
        # 3. 精确率-召回率曲线
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall_vals, precision_vals, color='blue', lw=2)
        axes[1, 0].set_xlabel('召回率')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].set_title('精确率-召回率曲线')
        
        # 4. 概率分布 -- 可视化预测概率的分布
        axes[1, 1].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                       bins=20, alpha=0.7, label=['不及格', '及格'], color=['red', 'green'])
        axes[1, 1].set_xlabel('预测概率')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('预测概率分布')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if pdf_dir:
            save_plot_as_pdf(plt, "comprehensive_model_evaluation", pdf_dir)
        
        plt.show()