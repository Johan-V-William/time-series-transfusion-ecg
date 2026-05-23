import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import pearsonr

class ECGTransFormEvaluator:
    def __init__(self, y_true, y_pred, y_prob=None):
        """
        y_true: Mảng label thực tế (1D)
        y_pred: Mảng label mô hình dự đoán (1D)
        y_prob: Xác suất dự đoán để tính AUC (2D cho multi-class)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.classes = np.unique(self.y_true)

    def _get_specificity(self):
        specs = []
        for c in self.classes:
            # Coi class hiện tại là Positive, các class khác là Negative
            tn, fp, fn, tp = confusion_matrix(self.y_true == c, self.y_pred == c).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specs.append(specificity)
        return specs

    def evaluate(self):
        # 1. Tính toán
        acc_per_class = [accuracy_score(self.y_true == c, self.y_pred == c) for c in self.classes]
        sens_per_class = recall_score(self.y_true, self.y_pred, average=None)
        spec_per_class = self._get_specificity()
        f1_macro = f1_score(self.y_true, self.y_pred, average='macro')
        
        # 2. In Report
        print("="*50)
        print("🚀 ECGTransForm EVALUATION REPORT")
        print("="*50)
        
        print(f"[1] METRIC CƠ BẢN (Dải phân bố qua {len(self.classes)} classes):")
        print(f" - Accuracy    : {min(acc_per_class):.1%} -> {max(acc_per_class):.1%}")
        print(f" - Sensitivity : {min(sens_per_class):.1%} -> {max(sens_per_class):.1%}")
        print(f" - Specificity : {min(spec_per_class):.1%} -> {max(spec_per_class):.1%}\n")

        print("[2] METRIC NÂNG CAO:")
        print(f" - F1-macro    : {f1_macro:.1%}")
        
        if self.y_prob is not None:
            # Multi-class AUC
            auc = roc_auc_score(self.y_true, self.y_prob, multi_class="ovr")
            print(f" - AUC/ROC     : {auc:.3f}")
            
        r, _ = pearsonr(self.y_true, self.y_pred)
        print(f" - Correlation : {r:.3f}\n")
        
        # 3. Phân tích tự động
        print("💡 AUTO DIAGNOSTICS:")
        if min(sens_per_class) < 0.5:
            print(" ⚠️ Mất cân bằng dữ liệu nghiêm trọng! Sensitivity đáy quá thấp.")
            print(" -> Suggest: Dùng Focal Loss, SMOTE hoặc check lại Data Imbalance.")
        else:
            print(" ✅ Các nhóm class học khá đều.")
        print("="*50)


# ==========================================
# CÁCH CHẠY THỬ (TEST)
# ==========================================
if __name__ == "__main__":
    # Giả lập dữ liệu test gồm 3 class rối loạn nhịp tim (0, 1, 2)
    np.random.seed(42)
    y_true_mock = np.random.choice([0, 1, 2], size=1000, p=[0.7, 0.2, 0.1]) # Imbalanced data
    
    # Mô hình dự đoán (Cố tình làm cho class thiểu số bị đoán sai)
    y_pred_mock = y_true_mock.copy()
    y_pred_mock[y_true_mock == 2] = np.random.choice([0, 2], size=(y_true_mock == 2).sum(), p=[0.7, 0.3])
    
    # Xác suất (Softmax output)
    y_prob_mock = np.random.rand(1000, 3)
    y_prob_mock = y_prob_mock / y_prob_mock.sum(axis=1, keepdims=True)

    # Chạy class
    evaluator = ECGTransFormEvaluator(y_true_mock, y_pred_mock, y_prob_mock)
    evaluator.evaluate()
