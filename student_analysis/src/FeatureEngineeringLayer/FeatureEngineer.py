import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class FeatureEngineer:
    """特征工程器"""

    def __init__(self):
        # 初始化标准化器和标签编码器
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_advanced_features(self, df):
        """创建复合特征"""
        df_processed = df.copy()
        
        # 1. 学业历史深度特征
        df_processed['academic_stability'] = (
            (df_processed['failures'] == 0).astype(int) * 2 +
            (df_processed['absences'] < 5).astype(int)
        )
        
        df_processed['study_consistency'] = (
            df_processed['studytime'] * (5 - df_processed['goout'])
        )
        
        # 2. 家庭支持综合指标
        df_processed['family_education_capital'] = (
            df_processed['Medu'] * 1.5 + 
            df_processed['Fedu'] * 1.2 +
            df_processed['famsup'].map({'yes': 2, 'no': 0})
        )
        
        df_processed['parental_involvement'] = (
            (df_processed['Mjob'] != 'at_home').astype(int) +
            (df_processed['Fjob'] != 'at_home').astype(int) +
            df_processed['famsup'].map({'yes': 1, 'no': 0})
        )
        
        # 3. 学习资源与环境
        df_processed['learning_environment'] = (
            df_processed['internet'].map({'yes': 2, 'no': 0}) +
            df_processed['higher'].map({'yes': 2, 'no': 0}) +
            (df_processed['traveltime'] <= 2).astype(int)
        )
        
        # 4. 行为模式特征
        df_processed['healthy_lifestyle'] = (
            df_processed['health'] + 
            (5 - df_processed['Dalc']) + 
            (5 - df_processed['Walc'])
        )
        
        df_processed['social_academic_balance'] = (
            df_processed['studytime'] * 2 - df_processed['goout']
        )
        
        # 5. 风险预警特征
        df_processed['risk_indicator'] = (
            df_processed['failures'] * 3 +
            (df_processed['absences'] > 10).astype(int) * 2 +
            (df_processed['studytime'] == 1).astype(int) * 2
        )
        
        # 6. 交互特征
        df_processed['edu_support_interaction'] = (
            df_processed['Medu'] * df_processed['famsup'].map({'yes': 1, 'no': 0})
        )
        
        df_processed['study_absence_ratio'] = (
            df_processed['studytime'] / (df_processed['absences'] + 1)
        )
        
        return df_processed

    def encode_categorical_features(self, df):
        """编码分类特征"""
        categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                              'Mjob', 'Fjob', 'reason', 'guardian', 
                              'schoolsup', 'famsup', 'paid', 'activities', 
                              'nursery', 'higher', 'internet', 'romantic']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                # 对分类特征进行编码
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def handle_imbalanced_data(self, X, y, strategy='smote'):
        """处理类别不平衡问题"""
        
        if strategy == 'smote':
            # 使用SMOTE重采样
            # 过采样 minority class
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif strategy == 'undersample':
            # 欠采样 majority class
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        else:
            # 不进行重采样
            X_resampled, y_resampled = X, y
            
        print(f"重采样后类别分布: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled
    
    # handle_imbalance 默认为 True -> 代表进行重采样（不传False都是重采样）
    def prepare_enhanced_features(self, df, handle_imbalance=True):
        """完整特征准备流程"""
        # 创建复合特征
        df = self.create_advanced_features(df)
        # 编码分类特征
        df = self.encode_categorical_features(df)
        
        # 选择最终特征集
        base_features = [
            'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
        ]
        
        advanced_features = [
            'academic_stability', 'study_consistency', 'family_education_capital',
            'parental_involvement', 'learning_environment', 'healthy_lifestyle',
            'social_academic_balance', 'risk_indicator', 'edu_support_interaction',
            'study_absence_ratio'
        ]
        
        categorical_features = [col for col in ['school', 'sex', 'address', 'famsize', 
                                              'Pstatus', 'Mjob', 'Fjob', 'reason', 
                                              'guardian'] if col in df.columns]
        
        feature_columns = base_features + advanced_features + categorical_features
        X = df[feature_columns]
        y = df['pass']
        
        # 处理类别不平衡
        if handle_imbalance:
            # handle_imbalance为True时，进行SMOTE重采样
            X, y = self.handle_imbalanced_data(X, y, strategy='smote')
        
        return X, y, feature_columns