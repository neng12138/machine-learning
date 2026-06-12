from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class ComparisonModel:
    """模型比较器"""

    def __init__(self):
        self.model = None
        self.best_model_name = ""  # 存储最佳模型名称
        
    def train_ensemble_models(self, X_train, y_train):
        """训练多种集成模型并选择最佳"""
        
        models = {
            # Random Forest -- 随机森林
            'RandomForest': RandomForestClassifier(random_state=42),
            # Gradient Boosting -- 梯度提升
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            # XGBoost -- 极端梯度提升（eXtreme Gradient Boosting）
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            # SVM -- 支持向量机
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # 定义每个模型的超参数网格
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': [None, 'balanced']
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        best_score = 0
        best_model = None
        best_model_name = ""
        
        for name, model in models.items():
            print(f"\n🔧 调优 {name}...")
            grid_search = GridSearchCV(
                model, param_grids[name], 
                cv=5, scoring='f1', n_jobs=-1, verbose=0
            )
            # 训练模型 - 重采样的训练集
            grid_search.fit(X_train, y_train)
            
            # 评估模型 - 重采样的验证集
            score = grid_search.best_score_
            print(f"✅ {name} 最佳F1分数: {score:.4f}")
            
            # 更新最佳模型
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = name
        
        print(f"\n🎯 选择最佳对比模型: {best_model_name} (F1: {best_score:.4f})")
        self.model = best_model
        self.best_model_name = best_model_name  # 保存最佳模型名称
        return self.model # 返回最佳模型