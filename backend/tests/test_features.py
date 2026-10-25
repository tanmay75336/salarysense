import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from fastapi.testclient import TestClient

from main import app
from ml.preprocessing import DataPreprocessor, MultiLabelEncoder
from ml.training import (
    ModelTrainer, 
    predict_salary, 
    find_similar_profiles, 
    explore_career_scenarios
)
from ml.registry import ModelRegistry
from services.dataset import (
    validate_dataset, 
    normalize_column_names, 
    load_csv_file,
    REQUIRED_COLUMNS
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def clean_dataset():
    """Create a multi-row dataset for robust testing."""
    records = []
    job_titles = ['Software Engineer', 'Senior Software Engineer', 'Lead Engineer', 'Principal Engineer']
    edu_levels = ['Bachelor', 'Master', 'PhD']
    countries = ['United States', 'Germany', 'India', 'Canada']
    
    for i in range(40):
        yoe = round(1.0 + (i * 0.4), 1)
        job = job_titles[i % len(job_titles)]
        edu = edu_levels[i % len(edu_levels)]
        country = countries[i % len(countries)]
        skills = ['Git', 'SQL']
        if i % 2 == 0:
            skills.append('Docker')
        if i % 3 == 0:
            skills.append('AWS')
        if i % 4 == 0:
            skills.append('Kubernetes')
            
        langs = ['Python']
        if i % 2 == 0:
            langs.append('JavaScript')
        if i % 3 == 0:
            langs.append('Go')
            
        base_salary = 60000 + (yoe * 12000) + (15000 if edu == 'Master' else (30000 if edu == 'PhD' else 0))
        
        records.append({
            'years_experience': yoe,
            'education_level': edu,
            'job_title': job,
            'skills': json.dumps(skills),
            'programming_languages': json.dumps(langs),
            'country': country,
            'annual_salary_usd': float(base_salary)
        })
        
    return pd.DataFrame(records)


class TestCandidateMLPipeline:
    def test_all_five_candidate_models_trained_and_stored(self, clean_dataset):
        trainer = ModelTrainer(random_state=42)
        results, training_run = trainer.train_all(clean_dataset, tune_hyperparameters=False)
        
        # Verify 5 distinct algorithms were trained
        assert len(results) == 5
        algorithms = {r.algorithm for r in results}
        expected_algos = {
            'linear_regression', 
            'ridge_regression', 
            'decision_tree', 
            'random_forest', 
            'gradient_boosting'
        }
        assert algorithms == expected_algos
        
        # Verify candidate models dictionary is populated
        assert len(trainer.candidate_models_) == 5
        for algo in expected_algos:
            assert algo in trainer.candidate_models_
            assert hasattr(trainer.candidate_models_[algo], 'predict')

    def test_selection_strictly_by_validation_mae(self, clean_dataset):
        trainer = ModelTrainer(random_state=42)
        results, training_run = trainer.train_all(clean_dataset, tune_hyperparameters=False)
        
        # The selected model must match the minimum validation MAE
        best_result = min(results, key=lambda r: r.validation_metrics.mae)
        assert training_run.selected_model == best_result.algorithm
        assert "Validation MAE" in training_run.selection_criteria
        assert trainer.best_model_name_ == best_result.algorithm

    def test_similar_profiles_excludes_target_salary(self, clean_dataset):
        trainer = ModelTrainer(random_state=42)
        trainer.train_all(clean_dataset, tune_hyperparameters=False)
        
        # Target column must NOT be in feature names
        assert trainer.preprocessor.target_column_ not in trainer.preprocessor.feature_names_
        
        # Test finding similar profiles
        input_data = {
            'years_experience': 3.0,
            'education_level': 'Bachelor',
            'job_title': 'Software Engineer',
            'skills': json.dumps(['Git', 'SQL', 'Docker']),
            'programming_languages': json.dumps(['Python', 'JavaScript']),
            'country': 'United States'
        }
        
        similar = find_similar_profiles(
            trainer.nn_model_,
            trainer.benchmark_df_,
            trainer.preprocessor,
            input_data,
            top_n=5
        )
        
        assert len(similar) <= 5
        assert len(similar) > 0
        for item in similar:
            assert 'profile' in item
            assert 'benchmark_salary' in item
            assert item['benchmark_salary'] > 0
            assert 'similarity_score' in item
            assert 0.0 <= item['similarity_score'] <= 100.0

    def test_career_scenario_explorer(self, clean_dataset):
        trainer = ModelTrainer(random_state=42)
        trainer.train_all(clean_dataset, tune_hyperparameters=False)
        
        input_data = {
            'years_experience': 2.0,
            'education_level': 'Bachelor',
            'job_title': 'Software Engineer',
            'skills': ['Git', 'SQL'],
            'programming_languages': ['Python'],
            'country': 'United States'
        }
        
        scenarios = explore_career_scenarios(
            trainer.best_model_,
            trainer.candidate_models_.get('decision_tree'),
            trainer.preprocessor,
            input_data
        )
        
        assert len(scenarios) >= 3
        for s in scenarios:
            assert 'title' in s
            assert 'new_salary' in s
            assert 'delta' in s
            assert 'delta_pct' in s
            assert s['new_salary'] > 0

    def test_salary_drivers_aggregation(self, clean_dataset):
        trainer = ModelTrainer(random_state=42)
        trainer.train_all(clean_dataset, tune_hyperparameters=False)
        
        drivers = trainer.feature_importances_
        assert 'categories' in drivers
        assert 'top_drivers' in drivers
        assert len(drivers['categories']) > 0
        assert len(drivers['top_drivers']) > 0
        # Category percentages should sum close to 100%
        total_pct = sum(drivers['categories'].values())
        assert 95.0 <= total_pct <= 105.0


class TestDatasetValidationAndAliases:
    def test_column_alias_normalization(self):
        alias_data = pd.DataFrame({
            'experience_years': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
            'degree': ['Bachelor'] * 11,
            'role': ['Software Engineer'] * 11,
            'skill_set': [json.dumps(['Git'])] * 11,
            'languages': [json.dumps(['Python'])] * 11,
            'location': ['United States'] * 11,
            'salary': [70000 + i * 5000 for i in range(11)]
        })
        
        norm_df, mapping, missing = normalize_column_names(alias_data)
        assert len(missing) == 0
        for col in REQUIRED_COLUMNS:
            assert col in norm_df.columns
            
        result = validate_dataset(alias_data)
        assert result.is_valid
        assert result.accepted_rows == 11
        assert result.total_input_rows == 11
        assert result.duplicate_rows == 0
        assert result.rejected_rows == 0

    def test_diagnostics_and_duplicate_tracking(self, clean_dataset):
        df_with_dups = pd.concat([clean_dataset, clean_dataset.iloc[:5]], ignore_index=True)
        # Add 2 invalid rows
        invalid_row1 = clean_dataset.iloc[0].to_dict()
        invalid_row1['years_experience'] = -5
        invalid_row2 = clean_dataset.iloc[1].to_dict()
        invalid_row2['annual_salary_usd'] = -100
        
        df_messy = pd.concat([
            df_with_dups, 
            pd.DataFrame([invalid_row1, invalid_row2])
        ], ignore_index=True)
        
        result = validate_dataset(df_messy)
        assert not result.is_valid  # Has errors due to negative experience & salary
        assert result.total_input_rows == len(df_messy)
        assert result.duplicate_rows == 5
        assert result.rejected_rows == 2
        assert result.excluded_reasons['negative_or_invalid_experience'] >= 1
        assert result.excluded_reasons['non_positive_or_invalid_salary'] >= 1

    def test_multilabel_comma_and_python_eval(self):
        encoder = MultiLabelEncoder('skills', top_k=5)
        df = pd.DataFrame({
            'skills': [
                "['Python', 'Django']",           # python literal string
                '["Python", "FastAPI"]',          # json string
                'Docker, Kubernetes, Helm',       # comma-separated string
                ['Python', 'AWS'],                # native list
            ]
        })
        encoder.fit(df)
        transformed = encoder.transform(df)
        assert 'skills_Python' in transformed.columns
        assert transformed.shape[0] == 4


class TestAPIEndpoints:
    def test_health_check(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data['status'] == "healthy"
        assert 'active_model' in data
        assert 'groq_available' in data

    def test_prediction_endpoint_with_linear_baseline(self, client):
        payload = {
            'years_experience': 4.0,
            'education_level': 'Bachelor',
            'job_title': 'Software Engineer',
            'skills': ['Git', 'SQL', 'Docker'],
            'programming_languages': ['Python', 'JavaScript'],
            'country': 'United States'
        }
        res = client.post("/api/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data['predicted_salary'] > 0
        assert data['currency'] == 'USD'
        assert data['model_name']
        assert 'baseline_salary' in data
        if data['baseline_salary'] is not None:
            assert data['baseline_salary'] > 0

    def test_insights_similar_profiles_endpoint(self, client):
        payload = {
            'years_experience': 4.0,
            'education_level': 'Bachelor',
            'job_title': 'Software Engineer',
            'skills': ['Git', 'SQL', 'Docker'],
            'programming_languages': ['Python'],
            'country': 'United States'
        }
        res = client.post("/api/insights/similar-profiles", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert 'profiles' in data
        assert len(data['profiles']) > 0
        assert 'benchmark_count' in data
        assert 'disclosure' in data

    def test_insights_scenarios_endpoint(self, client):
        payload = {
            'years_experience': 3.0,
            'education_level': 'Bachelor',
            'job_title': 'Software Engineer',
            'skills': ['Git', 'SQL'],
            'programming_languages': ['Python'],
            'country': 'United States'
        }
        res = client.post("/api/insights/scenarios", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert 'base_salary' in data
        assert 'scenarios' in data
        assert len(data['scenarios']) >= 3

    def test_insights_salary_drivers_endpoint(self, client):
        res = client.get("/api/insights/salary-drivers")
        assert res.status_code == 200
        data = res.json()
        assert 'categories' in data
        assert 'top_drivers' in data
        assert 'model_source' in data

    def test_csv_upload_validation_endpoint(self, client, clean_dataset):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', encoding='utf-8') as f:
            clean_dataset.to_csv(f.name, index=False)
            tmp_path = f.name
            
        try:
            with open(tmp_path, 'rb') as f:
                res = client.post(
                    "/api/dataset/upload/validate",
                    files={"file": ("dataset.csv", f, "text/csv")}
                )
            assert res.status_code == 200
            data = res.json()
            assert data['is_valid'] is True
            assert data['total_input_rows'] == len(clean_dataset)
            assert data['accepted_rows'] == len(clean_dataset)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
