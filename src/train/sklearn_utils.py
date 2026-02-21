#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Sklearn Training Utilities
================================================================================
本模組提供共用的機器學習訓練、評估與類別平衡輔助函式，
被 train_sklearn_classifier.py 與 train_rolling_hgb.py 等腳本所引用。
================================================================================
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.inspection import permutation_importance


def get_positive_proba(model, X, positive_label=1) -> tuple:
    """
    取得預測為 positive_label (預設為 1) 的機率陣列。
    避免依賴 hard-coded 的 [:, 1]，改由 model.classes_ 動態尋找。
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"模型 {type(model).__name__} 不支援 predict_proba()")
        
    proba_all = model.predict_proba(X)
    classes = list(model.classes_)
    
    if positive_label not in classes:
        raise ValueError(f"標籤 {positive_label} 不存在於 model.classes_ {classes} 中。")
        
    pos_idx = classes.index(positive_label)
    return proba_all[:, pos_idx], classes, pos_idx


def apply_class_balancing(df_train, balance_method, seed):
    """根據 balance_method 處理訓練集的 Class Imbalance"""
    if balance_method == 'undersample_50_50':
        pos_df = df_train[df_train['y'] == 1]
        neg_df = df_train[df_train['y'] == 0]
        min_len = min(len(pos_df), len(neg_df))
        if min_len == 0:
            return df_train
            
        pos_sample = pos_df.sample(n=min_len, random_state=seed)
        neg_sample = neg_df.sample(n=min_len, random_state=seed)
        
        # 確保順序不被打亂或者重排
        balanced_df = pd.concat([pos_sample, neg_sample]).sort_index()
        print(f"\n⚖️  [Undersample 50/50] 重新取樣後 Train Size: {len(balanced_df)} (Pos: {len(pos_sample)}, Neg: {len(neg_sample)})")
        return balanced_df
        
    return df_train


def get_model(model_name, balance_method, seed):
    """回傳指定模型與是否需要在 fit() 中使用 sample_weight"""
    class_weight = 'balanced' if balance_method == 'class_weight_balanced' else None
    
    if model_name == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                       random_state=seed, class_weight=class_weight, n_jobs=-1)
        return model, False # RF built-in handles class_weight

    elif model_name == 'adaboost':
        # DecisionTreeClassifier 支援 class_weight
        base = DecisionTreeClassifier(max_depth=2, class_weight=class_weight, random_state=seed)
        model = AdaBoostClassifier(estimator=base, n_estimators=100, random_state=seed)
        return model, False

    elif model_name == 'hgb':
        # HistGradientBoostingClassifier 雖然不直接支援 class_weight='balanced'
        # 在 sklearn 中可以改由 class_weight parameter (在 1.3+) 或是使用 fit 傳遞 sample_weight
        try:
            model = HistGradientBoostingClassifier(max_iter=100, max_depth=10, 
                                                   random_state=seed, class_weight=class_weight)
            return model, False
        except TypeError:
            # Fallback for older scikit-learn versions
            model = HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=seed)
            return model, (class_weight == 'balanced')
            
    else:
        raise ValueError(f"不支援的模型種類: {model_name}")


def calc_metrics(y_true, y_proba, y_pred, prefix="Overall"):
    """計算並回傳驗證集的各種指標"""
    metrics = {}
    
    # 避免 y_true 全 0 或全 1 導致 auc 失敗
    has_mixed_classes = len(np.unique(y_true)) > 1
    
    metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['F1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_proba)) if has_mixed_classes else None
    metrics['PR-AUC'] = float(average_precision_score(y_true, y_proba)) if has_mixed_classes else None
    
    metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Precision@k (Top 1%, 5%, 10%)
    sort_idx = np.argsort(y_proba)[::-1]
    sorted_y_true = np.array(y_true)[sort_idx]
    
    for k_pct in [0.01, 0.05, 0.10]:
        k = max(1, int(len(y_true) * k_pct))
        top_k_y_true = sorted_y_true[:k]
        metrics[f'Precision@{int(k_pct*100)}%'] = float(np.mean(top_k_y_true))
        
    # Threshold sweep
    metrics['Threshold Sweep'] = {}
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_th = (y_proba >= th).astype(int)
        metrics['Threshold Sweep'][f'Threshold={th}'] = {
            'Precision': float(precision_score(y_true, y_pred_th, zero_division=0)),
            'Recall': float(recall_score(y_true, y_pred_th, zero_division=0)),
            'F1': float(f1_score(y_true, y_pred_th, zero_division=0))
        }
        
    return metrics


def get_feature_importances(model, model_name, X_val, y_val, feature_cols):
    """計算並回傳特徵重要性"""
    importances_dict = {}
    print("\n🔍 正在計算 Feature Importances (Top 30)...")
    
    if model_name == 'rf':
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in indices[:30]:
                importances_dict[feature_cols[i]] = float(importances[i])
        except AttributeError:
            pass
    
    # 對其他模型使用 permutation importance (針對 Validation Subset 取樣以求效率)
    if not importances_dict:
        n_samples = min(50000, len(X_val))
        idx = np.random.choice(len(X_val), n_samples, replace=False)
        X_sub = X_val.iloc[idx]
        y_sub = y_val.iloc[idx]
        
        result = permutation_importance(model, X_sub, y_sub, n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]
        
        for i in indices[:30]:
            importances_dict[feature_cols[i]] = float(importances[i])
            
    return importances_dict
