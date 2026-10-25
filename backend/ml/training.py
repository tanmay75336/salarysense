import json
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator
import joblib

from .preprocessing import DataPreprocessor


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mae: float
    mse: float
    rmse: float
    r2: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> 'ModelMetrics':
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return cls(mae=mae, mse=mse, rmse=rmse, r2=r2)


@dataclass
class TrainingResult:
    """Results from a single candidate model training run."""
    model_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    validation_metrics: ModelMetrics
    test_metrics: Optional[ModelMetrics] = None
    cv_scores: Optional[Dict[str, List[float]]] = None
    training_time_seconds: float = 0.0
    n_training_samples: int = 0
    n_test_samples: int = 0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['validation_metrics'] = self.validation_metrics.to_dict()
        if self.test_metrics:
            d['test_metrics'] = self.test_metrics.to_dict()
        return d


@dataclass
class TrainingRun:
    """Complete training run metadata."""
    version: str
    dataset_name: str
    dataset_rows: int
    dataset_hash: str
    feature_schema: List[str]
    train_test_split: Dict[str, Any]
    timestamp: str
    results: List[TrainingResult]
    selected_model: Optional[str] = None
    selection_criteria: str = ""
    status: str = "completed"
    candidate_algorithms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['results'] = [r.to_dict() if isinstance(r, TrainingResult) else r for r in self.results]
        return d


class ModelTrainer:
    """Train and evaluate 5 candidate regression models with identical preprocessing & splits."""
    
    ALGORITHMS = {
        'linear_regression': {
            'name': 'Multiple Linear Regression',
            'description': 'Interpretable linear baseline showing foundational feature coefficients.',
            'estimator': LinearRegression,
            'param_grid': {},
            'default_params': {}
        },
        'ridge_regression': {
            'name': 'Ridge Regression (L2)',
            'description': 'L2 regularized linear model preventing coefficient inflation on correlated features.',
            'estimator': Ridge,
            'param_grid': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'default_params': {'alpha': 1.0, 'random_state': 42}
        },
        'decision_tree': {
            'name': 'Decision Tree Regressor',
            'description': 'Non-linear tree partitioning feature space into segmented career decisions.',
            'estimator': DecisionTreeRegressor,
            'param_grid': {'max_depth': [3, 5, 8], 'min_samples_split': [2, 5]},
            'default_params': {'max_depth': 5, 'random_state': 42}
        },
        'random_forest': {
            'name': 'Random Forest Regressor',
            'description': 'Bootstrap ensemble of trees used for primary estimation and feature importance driver analysis.',
            'estimator': RandomForestRegressor,
            'param_grid': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
            'default_params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting Regressor',
            'description': 'Sequential gradient boosted trees optimizing residual errors for high predictive precision.',
            'estimator': GradientBoostingRegressor,
            'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]},
            'default_params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
        }
    }
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2, cv_folds: int = 5):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.preprocessor: Optional[DataPreprocessor] = None
        self.best_model_: Optional[BaseEstimator] = None
        self.best_model_name_: Optional[str] = None
        self.candidate_models_: Dict[str, BaseEstimator] = {}
        self.nn_model_: Optional[NearestNeighbors] = None
        self.benchmark_df_: Optional[pd.DataFrame] = None
        self.feature_importances_: Dict[str, Any] = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split dataset into train, validation, and untouched test sets with identical preprocessing."""
        preprocessor = DataPreprocessor(random_state=self.random_state)
        X_transformed = preprocessor.fit_transform(df)
        y = preprocessor.get_target(df)
        
        # 1. Held-out test set (strictly untouched during validation model selection)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_transformed, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # 2. Train vs Validation split from trainval
        val_proportion = self.test_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_proportion, random_state=self.random_state
        )
        
        self.preprocessor = preprocessor
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(
        self, 
        algo_key: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        tune_hyperparameters: bool = True
    ) -> TrainingResult:
        """Train a candidate algorithm with optional hyperparameter tuning on train split only."""
        import time
        start_time = time.time()
        
        algo_info = self.ALGORITHMS[algo_key]
        estimator_class = algo_info['estimator']
        default_params = algo_info['default_params'].copy()
        warnings_list = []
        
        n_samples = len(X_train)
        cv = min(self.cv_folds if n_samples < 1000 else 3, max(2, n_samples // 3))
        
        if tune_hyperparameters and algo_info['param_grid'] and n_samples >= 6:
            try:
                grid_search = GridSearchCV(
                    estimator_class(**default_params),
                    algo_info['param_grid'],
                    cv=cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=1
                )
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_scores = {
                    'mean_cv_mae': [float(abs(score)) for score in grid_search.cv_results_['mean_test_score']],
                    'std_cv_mae': [float(score) for score in grid_search.cv_results_['std_test_score']]
                }
            except Exception as e:
                warnings_list.append(f"Grid search tuning note: {e}. Used robust default parameters.")
                best_estimator = estimator_class(**default_params)
                best_estimator.fit(X_train, y_train)
                best_params = default_params
                cv_scores = None
        else:
            best_estimator = estimator_class(**default_params)
            best_estimator.fit(X_train, y_train)
            best_params = default_params
            cv_scores = None
            
        # Validation MAE evaluation (this determines model selection)
        y_val_pred = best_estimator.predict(X_val)
        val_metrics = ModelMetrics.from_predictions(y_val.values, y_val_pred)
        
        training_time = float(time.time() - start_time)
        
        return TrainingResult(
            model_name=algo_info['name'],
            algorithm=algo_key,
            hyperparameters=best_params,
            validation_metrics=val_metrics,
            cv_scores=cv_scores,
            training_time_seconds=training_time,
            n_training_samples=len(X_train),
            n_test_samples=0,
            warnings=warnings_list
        )
        
    def evaluate_on_test(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model on the held-out test set."""
        y_pred = model.predict(X_test)
        return ModelMetrics.from_predictions(y_test.values, y_pred)
        
    def select_best_model(self, results: List[TrainingResult]) -> Tuple[str, str]:
        """Strictly select primary model by lowest Validation MAE."""
        valid_results = [r for r in results if r.validation_metrics.mae < float('inf')]
        if not valid_results:
            valid_results = results
            
        sorted_results = sorted(
            valid_results,
            key=lambda r: (r.validation_metrics.mae, -r.validation_metrics.r2)
        )
        
        best = sorted_results[0]
        criteria = f"Selected strictly by lowest Validation MAE (${best.validation_metrics.mae:,.2f}; Validation R²: {best.validation_metrics.r2:.3f})"
        if len(sorted_results) > 1:
            runner_up = sorted_results[1]
            criteria += f" | Runner-up: {runner_up.model_name} (Val MAE: ${runner_up.validation_metrics.mae:,.2f})"
            
        return best.algorithm, criteria

    def compute_feature_drivers(self, rf_model: RandomForestRegressor, preprocessor: DataPreprocessor) -> Dict[str, Any]:
        """Aggregate Random Forest feature importances into readable category groups and top drivers."""
        feature_names = preprocessor.feature_names_
        importances = getattr(rf_model, 'feature_importances_', None)
        if importances is None or len(importances) != len(feature_names):
            return {'categories': {}, 'top_drivers': []}
            
        cat_groups = {
            'Years of Experience': 0.0,
            'Job Title': 0.0,
            'Education Level': 0.0,
            'Country / Region': 0.0,
            'Technical Skills': 0.0,
            'Programming Languages': 0.0,
            'Other': 0.0
        }
        
        raw_drivers = []
        for feat, imp in zip(feature_names, importances):
            imp_val = float(imp)
            raw_drivers.append({'feature': feat, 'importance': imp_val})
            if feat.startswith('years_experience'):
                cat_groups['Years of Experience'] += imp_val
            elif feat.startswith('job_title'):
                cat_groups['Job Title'] += imp_val
            elif feat.startswith('education_level'):
                cat_groups['Education Level'] += imp_val
            elif feat.startswith('country'):
                cat_groups['Country / Region'] += imp_val
            elif feat.startswith('skills'):
                cat_groups['Technical Skills'] += imp_val
            elif feat.startswith('programming_languages'):
                cat_groups['Programming Languages'] += imp_val
            else:
                cat_groups['Other'] += imp_val
                
        # Normalize category percentages
        total_imp = sum(cat_groups.values()) or 1.0
        normalized_categories = {
            k: round((v / total_imp) * 100, 2)
            for k, v in cat_groups.items() if v > 0
        }
        
        # Sort top drivers
        raw_drivers.sort(key=lambda x: x['importance'], reverse=True)
        top_drivers = [
            {
                'name': d['feature'].replace('_', ' ').title(),
                'feature_key': d['feature'],
                'importance_pct': round((d['importance'] / total_imp) * 100, 2)
            }
            for d in raw_drivers[:10]
        ]
        
        return {
            'categories': normalized_categories,
            'top_drivers': top_drivers
        }

    def train_all(self, df: pd.DataFrame, tune_hyperparameters: bool = True) -> Tuple[List[TrainingResult], TrainingRun]:
        """Train all 5 candidate algorithms, build NearestNeighbors index, and package driver analysis."""
        clean_df = df.drop_duplicates().copy()
        
        # 1. Prepare splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(clean_df)
        
        results: List[TrainingResult] = []
        trained_candidates: Dict[str, BaseEstimator] = {}
        
        # 2. Train each candidate on train split, evaluate on validation split
        for algo_key in self.ALGORITHMS:
            try:
                res = self.train_single_model(
                    algo_key, X_train, y_train, X_val, y_val, tune_hyperparameters=tune_hyperparameters
                )
                
                # Refit candidate on full train+val and evaluate on untouched test set
                algo_info = self.ALGORITHMS[algo_key]
                final_model = algo_info['estimator'](**res.hyperparameters)
                X_trainval = pd.concat([X_train, X_val])
                y_trainval = pd.concat([y_train, y_val])
                final_model.fit(X_trainval, y_trainval)
                
                test_metrics = self.evaluate_on_test(final_model, X_test, y_test)
                res.test_metrics = test_metrics
                res.n_test_samples = len(X_test)
                
                results.append(res)
                trained_candidates[algo_key] = final_model
            except Exception as e:
                warnings.warn(f"Failed to train algorithm {algo_key}: {e}")
                results.append(TrainingResult(
                    model_name=self.ALGORITHMS[algo_key]['name'],
                    algorithm=algo_key,
                    hyperparameters={},
                    validation_metrics=ModelMetrics(mae=float('inf'), mse=float('inf'), rmse=float('inf'), r2=-float('inf')),
                    warnings=[f"Training failed: {e}"]
                ))
                
        # 3. Model selection strictly by validation MAE
        best_algo, criteria = self.select_best_model(results)
        self.best_model_name_ = best_algo
        self.best_model_ = trained_candidates.get(best_algo)
        self.candidate_models_ = trained_candidates
        
        # 4. Fit NearestNeighbors index on preprocessed features (TARGET STRICTLY EXCLUDED)
        X_all = self.preprocessor.transform(clean_df)
        n_neighbors = min(10, len(clean_df))
        self.nn_model_ = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.nn_model_.fit(X_all)
        self.benchmark_df_ = clean_df.copy()
        
        # 5. Extract Feature Drivers from Random Forest
        rf_model = trained_candidates.get('random_forest')
        if rf_model and isinstance(rf_model, RandomForestRegressor):
            self.feature_importances_ = self.compute_feature_drivers(rf_model, self.preprocessor)
        else:
            self.feature_importances_ = {'categories': {}, 'top_drivers': []}
            
        # 6. Version and metadata
        import hashlib
        dataset_hash = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:8]
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_hash}"
        
        training_run = TrainingRun(
            version=version,
            dataset_name="salary_data.csv",
            dataset_rows=len(df),
            dataset_hash=dataset_hash,
            feature_schema=self.preprocessor.feature_names_,
            train_test_split={
                'test_size': self.test_size,
                'random_state': self.random_state,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test)
            },
            timestamp=datetime.now().isoformat(),
            results=results,
            selected_model=best_algo,
            selection_criteria=criteria,
            candidate_algorithms=list(self.ALGORITHMS.keys())
        )
        
        return results, training_run

    def save_model(self, path: str, training_run: TrainingRun):
        """Save model bundle, preprocessor, and metadata."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        base_path = path
        if not base_path.endswith('.joblib'):
            base_path = base_path + '.joblib'
            
        # Save primary model (for backwards compatibility)
        model_path = base_path.replace('.joblib', '_model.joblib')
        joblib.dump(self.best_model_, model_path)
        
        # Save full bundle including all candidates, NN index, and benchmark dataset
        bundle_path = base_path.replace('.joblib', '_bundle.joblib')
        joblib.dump({
            'primary_model': self.best_model_,
            'selected_algorithm': self.best_model_name_,
            'candidate_models': self.candidate_models_,
            'nearest_neighbors': self.nn_model_,
            'feature_importances': self.feature_importances_,
            'benchmark_dataset': self.benchmark_df_
        }, bundle_path)
        
        # Save preprocessor
        preprocessor_path = base_path.replace('.joblib', '_preprocessor.joblib')
        self.preprocessor.save(preprocessor_path)
        
        # Save metadata
        metadata_path = base_path.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(training_run.to_dict(), f, indent=2)


def predict_salary(model: BaseEstimator, preprocessor: DataPreprocessor, input_data: Dict[str, Any]) -> Tuple[float, str]:
    """Single input salary prediction."""
    df = pd.DataFrame([input_data])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    return float(max(0.0, pred)), "USD"


def find_similar_profiles(
    nn_model: NearestNeighbors,
    benchmark_df: pd.DataFrame,
    preprocessor: DataPreprocessor,
    input_data: Dict[str, Any],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Find Nearest Neighbors in feature space. Target salary is strictly excluded from distance computation.
    """
    df = pd.DataFrame([input_data])
    X = preprocessor.transform(df)
    
    n_k = min(top_n, len(benchmark_df))
    distances, indices = nn_model.kneighbors(X, n_neighbors=n_k)
    
    similar_list = []
    for dist, idx in zip(distances[0], indices[0]):
        row = benchmark_df.iloc[idx].to_dict()
        similarity_pct = round(max(0.0, 1.0 - float(dist)) * 100, 1)
        similar_list.append({
            'profile': {
                'years_experience': float(row.get('years_experience', 0)),
                'education_level': str(row.get('education_level', '')),
                'job_title': str(row.get('job_title', '')),
                'country': str(row.get('country', '')),
                'skills': row.get('skills', []),
                'programming_languages': row.get('programming_languages', []),
            },
            'benchmark_salary': float(row.get('annual_salary_usd', 0)),
            'similarity_score': similarity_pct,
            'disclosure': "Historical profile from training benchmark. Target salary excluded from similarity calculation."
        })
        
    return similar_list


def explore_career_scenarios(
    primary_model: BaseEstimator,
    dt_model: Optional[BaseEstimator],
    preprocessor: DataPreprocessor,
    base_input: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Evaluate career variations (experience growth, degree advancement, upskilling)
    using the primary model and decision tree insights.
    """
    base_pred, _ = predict_salary(primary_model, preprocessor, base_input)
    
    curr_exp = float(base_input.get('years_experience', 0))
    curr_edu = str(base_input.get('education_level', 'Bachelor'))
    curr_skills = base_input.get('skills', [])
    if isinstance(curr_skills, str):
        try:
            curr_skills = json.loads(curr_skills)
        except Exception:
            curr_skills = [s.strip() for s in curr_skills.split(',') if s.strip()]
            
    curr_langs = base_input.get('programming_languages', [])
    if isinstance(curr_langs, str):
        try:
            curr_langs = json.loads(curr_langs)
        except Exception:
            curr_langs = [s.strip() for s in curr_langs.split(',') if s.strip()]
            
    scenarios = []
    
    # 1. Experience +2 Years
    input_exp2 = dict(base_input)
    input_exp2['years_experience'] = curr_exp + 2
    pred_exp2, _ = predict_salary(primary_model, preprocessor, input_exp2)
    scenarios.append({
        'title': '+2 Years Experience',
        'type': 'experience',
        'description': f'Advance experience from {curr_exp} to {curr_exp + 2} years in current role',
        'new_salary': pred_exp2,
        'delta': pred_exp2 - base_pred,
        'delta_pct': round(((pred_exp2 - base_pred) / max(1.0, base_pred)) * 100, 1),
        'rule_note': 'Derived from continuous model weight progression'
    })
    
    # 2. Experience +4 Years
    input_exp4 = dict(base_input)
    input_exp4['years_experience'] = curr_exp + 4
    pred_exp4, _ = predict_salary(primary_model, preprocessor, input_exp4)
    scenarios.append({
        'title': '+4 Years Experience',
        'type': 'experience',
        'description': f'Advance experience from {curr_exp} to {curr_exp + 4} years',
        'new_salary': pred_exp4,
        'delta': pred_exp4 - base_pred,
        'delta_pct': round(((pred_exp4 - base_pred) / max(1.0, base_pred)) * 100, 1),
        'rule_note': 'Mid-to-senior seniority progression threshold'
    })
    
    # 3. Education Degree Advancement
    next_edu = None
    if curr_edu in ['High School']:
        next_edu = 'Bachelor'
    elif curr_edu in ['Bachelor']:
        next_edu = 'Master'
    elif curr_edu in ['Master']:
        next_edu = 'PhD'
        
    if next_edu:
        input_edu = dict(base_input)
        input_edu['education_level'] = next_edu
        pred_edu, _ = predict_salary(primary_model, preprocessor, input_edu)
        scenarios.append({
            'title': f'Degree Advancement ({next_edu})',
            'type': 'education',
            'description': f'Attain {next_edu} degree while maintaining current role',
            'new_salary': pred_edu,
            'delta': pred_edu - base_pred,
            'delta_pct': round(((pred_edu - base_pred) / max(1.0, base_pred)) * 100, 1),
            'rule_note': f'Evaluated via {next_edu} categorical feature encoding'
        })
        
    # 4. High-demand Cloud/DevOps Upskilling
    upskill_set = ['AWS', 'Docker', 'Kubernetes']
    new_skills = list(set(curr_skills + [s for s in upskill_set if s not in curr_skills]))
    input_skills = dict(base_input)
    input_skills['skills'] = json.dumps(new_skills)
    pred_skills, _ = predict_salary(primary_model, preprocessor, input_skills)
    scenarios.append({
        'title': 'Cloud & Container Upskilling (AWS, Docker, K8s)',
        'type': 'skills',
        'description': 'Add containerization and cloud infrastructure competencies',
        'new_salary': pred_skills,
        'delta': pred_skills - base_pred,
        'delta_pct': round(((pred_skills - base_pred) / max(1.0, base_pred)) * 100, 1),
        'rule_note': 'Tree split rule & multi-hot feature interaction impact'
    })
    
    # 5. Language Expansion (Python + Go / TypeScript)
    new_langs = list(set(curr_langs + ['Go', 'TypeScript']))
    input_langs = dict(base_input)
    input_langs['programming_languages'] = json.dumps(new_langs)
    pred_langs, _ = predict_salary(primary_model, preprocessor, input_langs)
    scenarios.append({
        'title': 'High-Concurrency Languages (Go & TypeScript)',
        'type': 'languages',
        'description': 'Expand systems & modern web capabilities',
        'new_salary': pred_langs,
        'delta': pred_langs - base_pred,
        'delta_pct': round(((pred_langs - base_pred) / max(1.0, base_pred)) * 100, 1),
        'rule_note': 'Feature coefficient and non-linear tree split contribution'
    })
    
    return scenarios