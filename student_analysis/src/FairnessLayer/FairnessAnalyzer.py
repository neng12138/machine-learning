from sklearn.metrics import  confusion_matrix, f1_score

class FairnessAnalyzer:
    """公平性分析器"""

    def __init__(self, sensitive_attributes):
        # 初始化敏感属性列表
        self.sensitive_attributes = sensitive_attributes
    
    def calculate_fairness_metrics(self, y_true, y_pred, sensitive_data):
        """计算公平性指标"""
        fairness_results = {}
        
        for attr in self.sensitive_attributes:
            if attr in sensitive_data.columns:
                groups = sensitive_data[attr].unique()
                group_metrics = {}
                
                for group in groups:
                    mask = (sensitive_data[attr] == group)
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    
                    if len(y_true_group) > 0:
                        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                        
                        # 计算各指标
                        group_metrics[group] = {
                            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
                            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                            'f1': f1_score(y_true_group, y_pred_group),
                            'sample_size': len(y_true_group),
                            'pass_rate': y_true_group.mean()
                        }
                
                fairness_results[attr] = group_metrics
        
        return fairness_results
    
    def generate_fairness_report(self, fairness_results):
        """生成公平性分析报告"""
        print("=" * 60)
        print("公平性分析报告-逻辑回归模型")
        print("=" * 60)
        
        for attr, metrics in fairness_results.items():
            print(f"\n敏感属性: {attr}")
            print("-" * 40)
            
            for group, values in metrics.items():
                print(f"  群体 {group}:")
                print(f"    - 样本数: {values['sample_size']}")
                print(f"    - 及格率: {values['pass_rate']:.3f}")
                print(f"    - 假阳性率(FPR): {values['fpr']:.3f}")
                print(f"    - 假阴性率(FNR): {values['fnr']:.3f}")
                print(f"    - F1分数: {values['f1']:.3f}")
            
            # 计算公平性差异
            fprs = [values['fpr'] for values in metrics.values()]
            fnrs = [values['fnr'] for values in metrics.values()]
            
            print(f"  FPR最大差异: {max(fprs) - min(fprs):.3f}")
            print(f"  FNR最大差异: {max(fnrs) - min(fnrs):.3f}")

