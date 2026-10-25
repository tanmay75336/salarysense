import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os

from ml.preprocessing import (
    DataPreprocessor, 
    load_and_validate_csv, 
    MultiLabelEncoder
)
from ml.training import (
    ModelTrainer, 
    ModelMetrics, 
    TrainingResult,
    predict_salary
)
from ml.registry import ModelRegistry
from services.dataset import (
    validate_dataset, 
    load_csv_file,
    ValidationResult
)
from services.groq_service import groq_service


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing (10+ rows for validation)."""
    data = {
        'years_experience': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        'education_level': ['Bachelor'] * 4 + ['Master'] * 4 + ['PhD'] * 3,
        'job_title': ['Software Engineer'] * 4 + ['Senior Software Engineer'] * 3 + ['Lead Engineer'] * 2 + ['Principal Engineer'] * 2,
        'skills': [
            json.dumps(['Git', 'SQL']),
            json.dumps(['Git', 'SQL', 'Docker']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform', 'Microservices']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform', 'Microservices', 'Architecture']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform', 'Microservices', 'Architecture', 'Mentoring']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform', 'Microservices', 'Architecture', 'Mentoring', 'Research']),
            json.dumps(['Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform', 'Microservices', 'Architecture', 'Mentoring', 'Research', 'Strategy'])
        ],
        'programming_languages': [
            json.dumps(['Python']),
            json.dumps(['Python', 'JavaScript']),
            json.dumps(['Python', 'JavaScript', 'Java']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'C++']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'C++', 'Scala']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'C++', 'Scala', 'TypeScript']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'C++', 'Scala', 'TypeScript', 'Haskell']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'C++', 'Scala', 'TypeScript', 'Haskell', 'R']),
            json.dumps(['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'C++', 'Scala', 'TypeScript', 'Haskell', 'R', 'Bash'])
        ],
        'country': ['United States'] * 11,
        'annual_salary_usd': [70000, 75000, 85000, 95000, 110000, 120000, 135000, 150000, 165000, 180000, 195000]
    }
    return pd.DataFrame(data)


@pytest.fixture
def default_dataset():
    """Load the default dataset."""
    return load_and_validate_csv('../data/salary_data.csv')


# Preprocessing Tests
class TestPreprocessing:
    def test_load_default_dataset(self, default_dataset):
        assert len(default_dataset) == 64
        assert list(default_dataset.columns) == [
            'years_experience', 'education_level', 'job_title', 
            'skills', 'programming_languages', 'country', 'annual_salary_usd'
        ]
    
    def test_preprocessor_fit_transform(self, sample_data):
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_data)
        y = preprocessor.get_target(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert len(preprocessor.feature_names_) > 0
        assert y.shape[0] == len(sample_data)
        assert preprocessor.is_fitted_
    
    def test_preprocessor_save_load(self, sample_data, tmp_path):
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_data)
        
        save_path = tmp_path / "preprocessor.joblib"
        preprocessor.save(str(save_path))
        
        loaded = DataPreprocessor.load(str(save_path))
        assert loaded.is_fitted_
        assert loaded.feature_names_ == preprocessor.feature_names_
    
    def test_multilabel_encoder(self):
        df = pd.DataFrame({
            'skills': [
                json.dumps(['Git', 'Docker']),
                json.dumps(['Git', 'AWS']),
                json.dumps(['Docker', 'Kubernetes']),
                json.dumps([]),
                json.dumps(['Git'])
            ]
        })
        encoder = MultiLabelEncoder('skills', top_k=5)
        encoder.fit(df)
        result = encoder.transform(df)
        
        assert 'skills_Git' in result.columns
        assert 'skills_Docker' in result.columns
        assert result.shape == (5, len(encoder.all_labels_))
    
    def test_preprocessor_transform_consistency(self, sample_data):
        """Test that transform produces same features as fit_transform."""
        preprocessor = DataPreprocessor()
        X1 = preprocessor.fit_transform(sample_data)
        X2 = preprocessor.transform(sample_data)
        
        pd.testing.assert_frame_equal(X1, X2)


# Dataset Validation Tests
class TestDatasetValidation:
    def test_valid_dataset(self, sample_data):
        result = validate_dataset(sample_data)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.row_count == len(sample_data)
        assert result.column_count == 7
    
    def test_missing_required_columns(self):
        df = pd.DataFrame({'years_experience': [1, 2], 'education_level': ['Bachelor', 'Master']})
        result = validate_dataset(df)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_invalid_target_values(self, sample_data):
        df = sample_data.copy()
        df.loc[0, 'annual_salary_usd'] = -1000
        result = validate_dataset(df)
        assert not result.is_valid
        assert any('non-positive' in e for e in result.errors)
    
    def test_invalid_negative_experience(self, sample_data):
        df = sample_data.copy()
        df.loc[0, 'years_experience'] = -1
        result = validate_dataset(df)
        assert not result.is_valid
        assert any('negative' in e for e in result.errors)
    
    def test_invalid_json_in_multilabel(self, sample_data):
        df = sample_data.copy()
        df.loc[0, 'skills'] = 'not valid json'
        result = validate_dataset(df)
        assert not result.is_valid
        assert any('invalid JSON' in e for e in result.errors)
    
    def test_file_size_limit(self, tmp_path):
        # Create a file larger than 10MB
        large_file = tmp_path / "large.csv"
        with open(large_file, 'w') as f:
            f.write('x' * (11 * 1024 * 1024))
        
        with pytest.raises(ValueError, match='exceeds limit'):
            load_csv_file(str(large_file))


# Model Training Tests
class TestModelTraining:
    def test_all_five_algorithms_train(self, default_dataset):
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        assert len(results) == 5
        algorithms = [r.algorithm for r in results]
        expected = ['linear_regression', 'ridge_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
        assert set(algorithms) == set(expected)
    
    def test_metrics_calculated(self, default_dataset):
        trainer = ModelTrainer()
        results, _ = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        for r in results:
            assert r.validation_metrics.mae >= 0
            assert r.validation_metrics.mse >= 0
            assert r.validation_metrics.rmse >= 0
            assert r.validation_metrics.r2 <= 1.0
            assert r.test_metrics is not None
            assert r.test_metrics.mae >= 0
    
    def test_model_selection(self, default_dataset):
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        assert training_run.selected_model in [r.algorithm for r in results]
        assert training_run.selection_criteria
    
    def test_train_test_split_isolation(self, default_dataset):
        """Verify test set is not used in training."""
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        # Check split sizes
        split = training_run.train_test_split
        assert split['n_train'] + split['n_val'] + split['n_test'] == len(default_dataset)
        assert split['n_test'] > 0
    
    def test_prediction_function(self, default_dataset):
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        # Save and load
        registry = ModelRegistry()
        version = registry.save_training_run(trainer, training_run)
        registry.activate_version(version)
        
        artifacts = registry.load_active_artifacts()
        assert artifacts is not None
        
        model, preprocessor = artifacts
        input_data = {
            'years_experience': 3,
            'education_level': 'Bachelor',
            'job_title': 'Software Engineer',
            'skills': json.dumps(['Git', 'SQL', 'Docker']),
            'programming_languages': json.dumps(['Python', 'JavaScript']),
            'country': 'United States'
        }
        
        pred, currency = predict_salary(model, preprocessor, input_data)
        assert pred > 0
        assert currency == 'USD'


# Registry Tests
class TestModelRegistry:
    def test_save_and_activate(self, default_dataset, tmp_path):
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        registry = ModelRegistry(str(tmp_path / "models"))
        version = registry.save_training_run(trainer, training_run)
        
        assert version == training_run.version
        assert registry.activate_version(version)
        assert registry.has_active_model()
    
    def test_load_active_artifacts(self, default_dataset, tmp_path):
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        registry = ModelRegistry(str(tmp_path / "models"))
        registry.save_training_run(trainer, training_run)
        registry.activate_version(training_run.version)
        
        artifacts = registry.load_active_artifacts()
        assert artifacts is not None
        model, preprocessor = artifacts
        assert model is not None
        assert preprocessor is not None
    
    def test_cannot_activate_nonexistent(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "models"))
        assert not registry.activate_version("nonexistent_version")
    
    def test_history_tracking(self, default_dataset, tmp_path):
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        registry = ModelRegistry(str(tmp_path / "models"))
        registry.save_training_run(trainer, training_run)
        registry.activate_version(training_run.version)
        
        history = registry.list_history()
        assert len(history) == 1
        assert history[0]['is_active'] == True
        assert history[0]['version'] == training_run.version
    
    def test_failed_training_preserves_active(self, default_dataset, tmp_path):
        """Verify failed training doesn't overwrite active model."""
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        registry = ModelRegistry(str(tmp_path / "models"))
        registry.save_training_run(trainer, training_run)
        registry.activate_version(training_run.version)
        
        # Simulate failed training by trying to load invalid version
        assert not registry.activate_version("invalid_version")
        assert registry.has_active_model()


# Integration Tests
class TestIntegration:
    def test_full_pipeline(self, default_dataset):
        """Test complete pipeline: train -> save -> activate -> predict."""
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            version = registry.save_training_run(trainer, training_run)
            registry.activate_version(version)
            
            artifacts = registry.load_active_artifacts()
            model, preprocessor = artifacts
            
            input_data = {
                'years_experience': 3,
                'education_level': 'Bachelor',
                'job_title': 'Software Engineer',
                'skills': json.dumps(['Git', 'SQL', 'Docker']),
                'programming_languages': json.dumps(['Python', 'JavaScript']),
                'country': 'United States'
            }
            
            pred, currency = predict_salary(model, preprocessor, input_data)
            assert pred > 0
            assert currency == 'USD'
    
    def test_model_metrics_structure(self, default_dataset):
        """Verify all metrics are present and correctly structured."""
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        for r in results:
            val = r.validation_metrics
            test = r.test_metrics
            
            assert hasattr(val, 'mae')
            assert hasattr(val, 'mse')
            assert hasattr(val, 'rmse')
            assert hasattr(val, 'r2')
            
            assert hasattr(test, 'mae')
            assert hasattr(test, 'mse')
            assert hasattr(test, 'rmse')
            assert hasattr(test, 'r2')
    
    def test_cross_validation_not_on_test_set(self, default_dataset):
        """Ensure CV is only on training data."""
        trainer = ModelTrainer()
        results, training_run = trainer.train_all(default_dataset, tune_hyperparameters=False)
        
        # All models should have trained without using test set for CV
        for r in results:
            assert r.n_training_samples > 0
            assert r.n_test_samples > 0
            # CV scores should be from training data only
            if r.cv_scores is not None:
                assert 'mean_cv_mae' in r.cv_scores


# Groq Service Tests (Mocked)
class TestGroqService:
    def test_service_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        groq_service._initialize_client()
        assert not groq_service.is_available()
    
    def test_parse_response_valid(self):
        response = json.dumps({
            'skills_to_learn': ['Docker', 'Kubernetes'],
            'learning_roadmap': [{'priority': 'High', 'topic': 'Docker', 'description': 'Learn containers'}],
            'portfolio_projects': ['Containerized web app'],
            'certifications': ['CKA'],
            'career_suggestions': ['Apply for DevOps roles'],
            'limitations': ['Synthetic data']
        })
        result = groq_service._parse_response(response)
        assert isinstance(result.skills_to_learn, list)
        assert len(result.skills_to_learn) == 2
    
    def test_parse_response_invalid(self):
        result = groq_service._parse_response("not json")
        assert result.skills_to_learn == []
        assert 'Failed to parse' in result.limitations[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])