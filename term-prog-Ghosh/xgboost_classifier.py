# xgboost_classifier.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import shap
import os


def load_data(embedding_path, label_path):
    X = np.load(embedding_path)
    y = np.load(label_path)
    return X, y


def train_xgboost(X_train, y_train, X_val, y_val):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
    }

    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    clf = GridSearchCV(xgb, param_grid, cv=3, verbose=1)
    clf.fit(X_train, y_train)

    print("Best Parameters:", clf.best_params_)
    return clf.best_estimator_


def evaluate_xgboost(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Pathogenic"])
    disp.plot()
    plt.title("XGBoost Confusion Matrix")
    plt.savefig("figures/xgboost_confusion_matrix.png")
    plt.show()


def explain_with_shap(model, X, feature_names=None):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig("figures/xgboost_feature_importance.png")
    plt.close()
