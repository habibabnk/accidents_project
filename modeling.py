import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import pickle
import os
from pathlib import Path

# Safe Streamlit imports with fallback
cache_data = None
cache_resource = None
try:
    import streamlit as st
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except Exception:
    st = None
    def cache_data(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    def cache_resource(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

class AccidentRiskModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.pipeline = None
        self.feature_columns = []
        self.categorical_features = []
        self.numerical_features = []
        self.is_trained = False
        
    def prepare_data(self, df, target_column='is_serious'):
        """Prepare data for modeling"""
        if df is None:
            raise ValueError("No data provided")
            
        # Filter to years with complete data
        df = df[df['year'] >= 2019].copy()  # Use more recent data for better quality
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_column])
        
        # Define feature columns
        potential_categorical = [
            'month', 'day_of_week', 'hour', 'department', 'lighting',
            'weather', 'intersection', 'collision_type', 'localization'
        ]
        
        potential_numerical = ['year']
        
        # Select available features
        self.categorical_features = [col for col in potential_categorical if col in df.columns]
        self.numerical_features = [col for col in potential_numerical if col in df.columns]
        self.feature_columns = self.categorical_features + self.numerical_features
        
        if not self.feature_columns:
            raise ValueError("No suitable features found for modeling")
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()
        
        # Clean categorical data
        for col in self.categorical_features:
            X[col] = X[col].astype(str).fillna('unknown')
            # Remove any problematic characters
            X[col] = X[col].str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        
        # Clean numerical data
        for col in self.numerical_features:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        print(f"Features available: {self.feature_columns}")
        print(f"Target distribution: {y.value_counts()}")
        
        return X, y
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ]
        )
        
        return self.preprocessor
    
    def create_model(self):
        """Create the model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X, y, test_size=0.2, time_split=True):
        """Train the model"""
        if time_split and 'year' in X.columns:
            # Time-based split to avoid leakage
            max_year = X['year'].max()
            split_year = max_year - 1
            
            X_train = X[X['year'] <= split_year]
            X_test = X[X['year'] > split_year]
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create preprocessor and model
        self.create_preprocessor()
        self.create_model()
        
        # Create full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_names': self.pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
        }
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"AUC: {metrics['auc']:.3f}")
        
        return metrics, X_train, X_test, y_train, y_test
    
    def predict_risk(self, conditions):
        """Predict risk for given conditions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Convert conditions to DataFrame
        input_df = pd.DataFrame([conditions])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 'unknown' if col in self.categorical_features else 0
        
        # Make prediction
        risk_proba = self.pipeline.predict_proba(input_df)[0, 1]
        risk_class = self.pipeline.predict(input_df)[0]
        
        return {
            'risk_probability': float(risk_proba),
            'risk_class': int(risk_class),
            'risk_percentage': float(risk_proba * 100)
        }
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.feature_columns = model_data['feature_columns']
        self.categorical_features = model_data['categorical_features']
        self.numerical_features = model_data['numerical_features']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

# Global model instance
_risk_model = None

def get_risk_model():
    """Get or create risk model instance"""
    global _risk_model
    if _risk_model is None:
        _risk_model = AccidentRiskModel()
    return _risk_model

@cache_resource
def train_risk_model(data):
    """Streamlit cached model training"""
    model = get_risk_model()
    
    try:
        X, y = model.prepare_data(data)
        metrics, X_train, X_test, y_train, y_test = model.train(X, y)
        return True, metrics, model
    except Exception as e:
        print(f"Model training failed: {e}")
        return False, str(e), None

def predict_accident_risk(conditions):
    """Predict accident risk for given conditions"""
    model = get_risk_model()
    return model.predict_risk(conditions)
