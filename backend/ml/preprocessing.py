import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Optional
import joblib
import warnings


import ast

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode multi-label columns (JSON arrays, Python lists, or comma-separated strings) as multi-hot features."""
    
    def __init__(self, column_name: str, top_k: int = 30):
        self.column_name = column_name
        self.top_k = top_k
        self.feature_names_: List[str] = []
        self.all_labels_: List[str] = []
        self.label_to_idx_: Dict[str, int] = {}
        
    def _parse_labels(self, val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple, np.ndarray)):
            return [str(item).strip() for item in val if str(item).strip()]
        if pd.isna(val) or val == '' or val == '[]':
            return []
        val_str = str(val).strip()
        if not val_str or val_str == '[]':
            return []
        
        # 1. Try standard JSON
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
            
        # 2. Try Python literal eval (e.g. "['Python', 'SQL']")
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError, TypeError):
            pass
            
        # 3. Fallback: comma-separated values (strip brackets/quotes if present)
        cleaned = val_str.strip('[](){}"\' ')
        if cleaned:
            return [s.strip().strip('\'"') for s in cleaned.split(',') if s.strip()]
        return []
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            col_data = X[self.column_name]
        else:
            col_data = pd.Series(X)
        
        all_labels = []
        for val in col_data:
            labels = self._parse_labels(val)
            all_labels.extend(labels)
        
        label_counts = pd.Series(all_labels).value_counts()
        self.all_labels_ = label_counts.head(self.top_k).index.tolist()
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.all_labels_)}
        self.feature_names_ = [f"{self.column_name}_{label}" for label in self.all_labels_]
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            col_data = X[self.column_name]
        else:
            col_data = pd.Series(X)
        
        n_samples = len(col_data)
        n_features = len(self.all_labels_)
        result = np.zeros((n_samples, n_features), dtype=np.float32)
        
        if n_features == 0:
            return pd.DataFrame(result, columns=self.feature_names_, index=col_data.index)
            
        for i, val in enumerate(col_data):
            labels = self._parse_labels(val)
            for label in labels:
                idx = self.label_to_idx_.get(label)
                if idx is not None:
                    result[i, idx] = 1.0
        
        return pd.DataFrame(result, columns=self.feature_names_, index=col_data.index)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class DataPreprocessor:
    """Complete preprocessing pipeline for salary prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.categorical_columns_ = ['education_level', 'job_title', 'country']
        self.numeric_columns_ = ['years_experience']
        self.multilabel_columns_ = ['skills', 'programming_languages']
        self.target_column_ = 'annual_salary_usd'
        self.is_fitted_ = False
        
    def _validate_dataframe(self, df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        """Validate and clean input dataframe."""
        df = df.copy()
        
        required_columns = self.categorical_columns_ + self.numeric_columns_ + self.multilabel_columns_
        if require_target:
            required_columns.append(self.target_column_)
        
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Serialize list-like columns before drop_duplicates
        for col in self.multilabel_columns_:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(list(x)) if isinstance(x, (list, tuple, np.ndarray)) else (str(x) if pd.notna(x) else "")
                )
        
        # Check for duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            warnings.warn(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Validate numeric columns
        for col in self.numeric_columns_:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                raise ValueError(f"Column '{col}' contains invalid numeric values")
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values")
        
        # Validate target
        if require_target:
            df[self.target_column_] = pd.to_numeric(df[self.target_column_], errors='coerce')
            if df[self.target_column_].isna().any():
                raise ValueError(f"Target column '{self.target_column_}' contains invalid values")
            if (df[self.target_column_] <= 0).any():
                raise ValueError(f"Target column '{self.target_column_}' contains non-positive values")
        
        # Validate categorical columns
        for col in self.categorical_columns_:
            df[col] = df[col].astype(str).str.strip()
            if df[col].str.len().eq(0).any():
                raise ValueError(f"Column '{col}' contains empty values")
        
        # Validate multilabel columns (ensure they're parseable or already lists)
        temp_encoder = MultiLabelEncoder('temp')
        for col in self.multilabel_columns_:
            for val in df[col]:
                if isinstance(val, list):
                    continue
                parsed = temp_encoder._parse_labels(val)
                # An empty list or valid parsed list is fine. If parsing returns empty from non-empty string, verify it's not corrupt.
                if not parsed and pd.notna(val) and str(val).strip() not in ['', '[]']:
                    raise ValueError(f"Column '{col}' contains unparseable multi-label values: {val}")
        
        return df
    
    def build_preprocessor(self) -> ColumnTransformer:
        """Build the column transformer for all feature types."""
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Multi-label pipelines
        skills_encoder = MultiLabelEncoder('skills', top_k=30)
        languages_encoder = MultiLabelEncoder('programming_languages', top_k=20)
        
        transformers = [
            ('numeric', numeric_pipeline, self.numeric_columns_),
            ('categorical', categorical_pipeline, self.categorical_columns_),
            ('skills', skills_encoder, ['skills']),
            ('languages', languages_encoder, ['programming_languages'])
        ]
        
        return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit the preprocessor on training data."""
        df = self._validate_dataframe(df, require_target=True)
        
        X = df.drop(columns=[self.target_column_])
        self.preprocessor_ = self.build_preprocessor()
        self.preprocessor_.fit(X)
        
        # Get feature names
        self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist()
        self.is_fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        df = self._validate_dataframe(df, require_target=False)
        X = df.drop(columns=[self.target_column_], errors='ignore')
        transformed = self.preprocessor_.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_names_, index=X.index)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Extract target variable."""
        df = self._validate_dataframe(df, require_target=True)
        return df[self.target_column_].copy()
    
    def save(self, path: str):
        """Save fitted preprocessor to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted preprocessor")
        joblib.dump({
            'preprocessor': self.preprocessor_,
            'feature_names': self.feature_names_,
            'categorical_columns': self.categorical_columns_,
            'numeric_columns': self.numeric_columns_,
            'multilabel_columns': self.multilabel_columns_,
            'target_column': self.target_column_,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted_
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load fitted preprocessor from disk."""
        data = joblib.load(path)
        obj = cls(random_state=data['random_state'])
        obj.preprocessor_ = data['preprocessor']
        obj.feature_names_ = data['feature_names']
        obj.categorical_columns_ = data['categorical_columns']
        obj.numeric_columns_ = data['numeric_columns']
        obj.multilabel_columns_ = data['multilabel_columns']
        obj.target_column_ = data['target_column']
        obj.is_fitted_ = data['is_fitted']
        return obj


def load_and_validate_csv(path: str) -> pd.DataFrame:
    """Load CSV with basic validation."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    if df.empty:
        raise ValueError("CSV file is empty")
    
    if len(df) < 10:
        warnings.warn(f"Dataset has only {len(df)} rows. Minimum 10 recommended for training.")
    
    return df