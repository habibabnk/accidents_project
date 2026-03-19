import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import pickle
import os
from pathlib import Path

try:
    import streamlit as st
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(func=None, **_):
        return func if func else lambda f: f


class AccidentRiskModel:
    def __init__(self):
        self.pipeline = None
        self.feature_columns = []
        self.categorical_features = []
        self.numerical_features = []
        self.is_trained = False
        self.classes_ = None

    def prepare_data(self, df, target_column='is_serious'):
        if df is None:
            raise ValueError("No data provided")

        df = df.copy()

        # Need both classes present
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        df = df.dropna(subset=[target_column])
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0).astype(int)

        class_counts = df[target_column].value_counts()
        if len(class_counts) < 2:
            raise ValueError(
                f"Target has only one class ({class_counts.index[0]}). "
                "Cannot train a binary classifier. "
                "Check that your data contains both serious and non-serious accidents."
            )

        potential_categorical = [
            'month', 'day_of_week', 'hour', 'department',
            'lighting', 'weather', 'intersection', 'collision_type', 'localization'
        ]
        potential_numerical = ['year']

        self.categorical_features = [c for c in potential_categorical if c in df.columns]
        self.numerical_features = [c for c in potential_numerical if c in df.columns]
        self.feature_columns = self.categorical_features + self.numerical_features

        if not self.feature_columns:
            raise ValueError("No usable feature columns found")

        X = df[self.feature_columns].copy()
        y = df[target_column].copy()

        for col in self.categorical_features:
            X[col] = X[col].astype(str).fillna('unknown')
        for col in self.numerical_features:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        return X, y

    def train(self, X, y):
        # Time-based split if year available, else random
        if 'year' in X.columns and X['year'].nunique() > 1:
            split_year = int(X['year'].max()) - 1
            X_train = X[X['year'] <= split_year]
            X_test  = X[X['year'] >  split_year]
            y_train = y.loc[X_train.index]
            y_test  = y.loc[X_test.index]
            # Fallback if test set too small
            if len(X_test) < 100 or y_test.nunique() < 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        preprocessor = ColumnTransformer([
            ('cat', cat_pipe, self.categorical_features),
            ('num', num_pipe, self.numerical_features),
        ])

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42, class_weight='balanced'
        )

        self.pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.classes_
        self.is_trained = True

        y_pred  = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        # Safe AUC
        if y_proba.shape[1] >= 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = 0.5

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        return metrics, X_train, X_test, y_train, y_test

    def predict_risk(self, conditions):
        if not self.is_trained:
            raise ValueError("Model not trained")

        row = pd.DataFrame([conditions])
        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = 'unknown' if col in self.categorical_features else 0

        proba = self.pipeline.predict_proba(row)

        # Handle case where only one class was seen at predict time
        if proba.shape[1] >= 2:
            risk = float(proba[0, 1])
        else:
            # Only class present — check which one
            cls = int(self.pipeline.predict(row)[0])
            risk = 1.0 if cls == 1 else 0.0

        return {
            'risk_probability': risk,
            'risk_class': int(self.pipeline.predict(row)[0]),
            'risk_percentage': risk * 100,
        }

    def get_feature_importance(self):
        if not self.is_trained:
            return None
        clf = self.pipeline.named_steps['clf']
        if not hasattr(clf, 'feature_importances_'):
            return None
        names = self.pipeline.named_steps['pre'].get_feature_names_out()
        return pd.DataFrame({
            'feature': names,
            'importance': clf.feature_importances_,
        }).sort_values('importance', ascending=False)


# ── Singletons ────────────────────────────────────────────────────────────────
_model = None

def get_risk_model():
    global _model
    if _model is None:
        _model = AccidentRiskModel()
    return _model


@cache_resource
def train_risk_model(data):
    model = get_risk_model()
    try:
        X, y = model.prepare_data(data)
        metrics, *_ = model.train(X, y)
        return True, metrics, model
    except Exception as e:
        return False, str(e), None


def predict_accident_risk(conditions):
    return get_risk_model().predict_risk(conditions)