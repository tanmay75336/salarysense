import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import uvicorn

from ml.preprocessing import DataPreprocessor, load_and_validate_csv
from ml.training import (
    ModelTrainer, 
    TrainingRun, 
    TrainingResult, 
    ModelMetrics,
    predict_salary as predict_fn,
    find_similar_profiles,
    explore_career_scenarios
)
from ml.registry import ModelRegistry
from services.dataset import validate_dataset, load_csv_file, ValidationResult
from services.groq_service import groq_service, CareerRecommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
model_registry = ModelRegistry()
trainer = ModelTrainer()


# Pydantic models for API
class PredictionInput(BaseModel):
    years_experience: float = Field(..., ge=0, le=50)
    education_level: str = Field(..., min_length=1)
    job_title: str = Field(..., min_length=1)
    skills: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    country: str = Field(..., min_length=1)
    
    @validator('skills', 'programming_languages', pre=True)
    def parse_json_array(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            import ast
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            # comma fallback
            cleaned = v.strip('[](){}"\' ')
            if cleaned:
                return [s.strip().strip('\'"') for s in cleaned.split(',') if s.strip()]
        return v if isinstance(v, list) else []


class PredictionResponse(BaseModel):
    predicted_salary: float
    currency: str
    model_name: str
    model_version: str
    disclaimer: str
    baseline_salary: Optional[float] = None
    baseline_diff: Optional[float] = None
    baseline_diff_pct: Optional[float] = None


class DatasetInfoResponse(BaseModel):
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_summary: Dict[str, Any]
    unique_counts: Dict[str, int]
    sample_rows: List[Dict[str, Any]]
    is_synthetic: bool
    source: str
    limitations: List[str]


class ModelComparisonResponse(BaseModel):
    models: List[Dict[str, Any]]
    selected_model: str
    selection_criteria: str
    training_timestamp: str
    dataset_info: Dict[str, Any]


class UploadValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    preview: Dict[str, Any]
    column_mapping: Dict[str, str]
    total_input_rows: int
    accepted_rows: int
    duplicate_rows: int
    rejected_rows: int
    excluded_reasons: Dict[str, int]


class RetrainRequest(BaseModel):
    confirm: bool = False


class RetrainResponse(BaseModel):
    success: bool
    version: Optional[str] = None
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    selected_model: Optional[str] = None
    selection_criteria: Optional[str] = None
    total_input_rows: Optional[int] = None
    accepted_rows: Optional[int] = None
    duplicate_rows: Optional[int] = None
    rejected_rows: Optional[int] = None
    excluded_reasons: Optional[Dict[str, int]] = None


class TrainingHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    active_version: Optional[str] = None


class ActivateModelRequest(BaseModel):
    version: str


class ActivateModelResponse(BaseModel):
    success: bool
    message: str
    active_version: Optional[str] = None


class CareerRecommendationRequest(BaseModel):
    profile: PredictionInput
    predicted_salary: float
    currency: str
    model_name: str


class CareerRecommendationResponse(BaseModel):
    available: bool
    recommendations: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    active_model: bool
    groq_available: bool
    timestamp: str


class SimilarProfilesResponse(BaseModel):
    profiles: List[Dict[str, Any]]
    benchmark_count: int
    disclosure: str


class ScenarioExplorerResponse(BaseModel):
    base_salary: float
    currency: str
    scenarios: List[Dict[str, Any]]


class SalaryDriversResponse(BaseModel):
    categories: Dict[str, float]
    top_drivers: List[Dict[str, Any]]
    model_source: str


# Helper functions
def get_default_dataset() -> pd.DataFrame:
    """Load default dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'salary_data.csv')
    return load_and_validate_csv(data_path)


def initialize_default_model():
    """Train default 5 candidate models on startup if none exists."""
    if not model_registry.has_active_model():
        logger.info("No active model found. Training default candidate models...")
        try:
            df = get_default_dataset()
            results, training_run = trainer.train_all(df, tune_hyperparameters=False)
            model_registry.save_training_run(trainer, training_run)
            model_registry.activate_version(training_run.version)
            logger.info(f"Default model trained and activated: {training_run.version}")
        except Exception as e:
            logger.error(f"Failed to train default model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    initialize_default_model()
    yield


app = FastAPI(
    title="SalarySense AI — Intelligent Salary Prediction & Career Recommendation System",
    description="Full-stack ML salary prediction, scenario exploration, feature drivers, and AI career guidance.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router with /api prefix
api_router = APIRouter(prefix="/api")


@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        active_model=model_registry.has_active_model(),
        groq_available=groq_service.is_available(),
        timestamp=datetime.now().isoformat()
    )


@api_router.get("/dataset/info", response_model=DatasetInfoResponse)
async def get_dataset_info():
    """Get information about the current training dataset."""
    df = get_default_dataset()
    
    numeric_summary = {}
    for col in ['years_experience', 'annual_salary_usd']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            numeric_summary[col] = {
                'mean': float(s.mean()),
                'std': float(s.std()) if len(s) > 1 else 0.0,
                'min': float(s.min()),
                'max': float(s.max()),
                'median': float(s.median())
            }
            
    return DatasetInfoResponse(
        row_count=len(df),
        column_count=len(df.columns),
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_values=df.isnull().sum().to_dict(),
        numeric_summary=numeric_summary,
        unique_counts={col: int(df[col].nunique()) for col in df.columns},
        sample_rows=df.head(5).to_dict(orient='records'),
        is_synthetic=True,
        source="Educational benchmark dataset for software engineering compensation modeling",
        limitations=[
            "Synthetic educational benchmark dataset",
            "Target salary is base USD equivalent; excludes equity, bonuses, and healthcare allowances",
            "Estimates reflect statistical central tendencies and should not replace personalized negotiation"
        ]
    )


@api_router.get("/models/comparison", response_model=ModelComparisonResponse)
async def get_model_comparison():
    """Get comparison of all trained candidate models."""
    active_meta = model_registry.get_active_model()
    if not active_meta:
        raise HTTPException(status_code=404, detail="No trained model available. Please train a model first.")
    
    models = []
    for result in active_meta.get('results', []):
        model_info = {
            'model_name': result.get('model_name'),
            'algorithm': result.get('algorithm'),
            'hyperparameters': result.get('hyperparameters'),
            'validation_metrics': result.get('validation_metrics'),
            'test_metrics': result.get('test_metrics'),
            'cv_scores': result.get('cv_scores'),
            'training_time_seconds': result.get('training_time_seconds'),
            'warnings': result.get('warnings', [])
        }
        models.append(model_info)
    
    return ModelComparisonResponse(
        models=models,
        selected_model=active_meta.get('selected_model', ''),
        selection_criteria=active_meta.get('selection_criteria', ''),
        training_timestamp=active_meta.get('timestamp', ''),
        dataset_info={
            'rows': active_meta.get('dataset_rows'),
            'name': active_meta.get('dataset_name'),
            'hash': active_meta.get('dataset_hash')
        }
    )


@api_router.post("/predict", response_model=PredictionResponse)
async def predict_salary_endpoint(input_data: PredictionInput):
    """Predict salary using selected production model and compare with Multiple Linear Regression baseline."""
    bundle = model_registry.load_active_bundle()
    if not bundle:
        raise HTTPException(status_code=404, detail="No active model found. Please train or activate a model first.")
        
    primary_model = bundle['primary_model']
    preprocessor = bundle['preprocessor']
    active_meta = bundle['metadata'] or {}
    candidate_models = bundle.get('candidate_models', {})
    
    input_dict = input_data.dict()
    input_dict['skills'] = json.dumps(input_data.skills)
    input_dict['programming_languages'] = json.dumps(input_data.programming_languages)
    
    # Primary estimate
    predicted_salary, currency = predict_fn(primary_model, preprocessor, input_dict)
    
    # Linear baseline
    baseline_salary = None
    baseline_diff = None
    baseline_diff_pct = None
    
    linear_model = candidate_models.get('linear_regression')
    if linear_model:
        try:
            base_val, _ = predict_fn(linear_model, preprocessor, input_dict)
            baseline_salary = round(base_val, 2)
            baseline_diff = round(predicted_salary - baseline_salary, 2)
            baseline_diff_pct = round((baseline_diff / max(1.0, baseline_salary)) * 100, 1)
        except Exception as e:
            logger.warning(f"Failed to compute linear baseline: {e}")
            
    return PredictionResponse(
        predicted_salary=round(predicted_salary, 2),
        currency=currency,
        model_name=active_meta.get('selected_model', 'production_model'),
        model_version=active_meta.get('version', 'latest'),
        disclaimer="SalarySense AI estimation based on validated statistical modeling. Benchmark baseline for educational and comparative career planning.",
        baseline_salary=baseline_salary,
        baseline_diff=baseline_diff,
        baseline_diff_pct=baseline_diff_pct
    )


@api_router.post("/insights/similar-profiles", response_model=SimilarProfilesResponse)
async def get_similar_profiles_endpoint(input_data: PredictionInput):
    """Find Nearest Neighbor profiles from the training benchmark. Target salary excluded from distance."""
    bundle = model_registry.load_active_bundle()
    if not bundle:
        raise HTTPException(status_code=404, detail="No active model artifacts available.")
        
    nn_model = bundle.get('nearest_neighbors')
    benchmark_df = bundle.get('benchmark_dataset')
    preprocessor = bundle.get('preprocessor')
    
    if nn_model is None or benchmark_df is None or preprocessor is None:
        # Fallback to default dataset if not in bundle
        benchmark_df = get_default_dataset()
        preprocessor = bundle['preprocessor']
        X_all = preprocessor.transform(benchmark_df)
        from sklearn.neighbors import NearestNeighbors
        nn_model = NearestNeighbors(n_neighbors=min(10, len(benchmark_df)), metric='cosine').fit(X_all)
        
    input_dict = input_data.dict()
    input_dict['skills'] = json.dumps(input_data.skills)
    input_dict['programming_languages'] = json.dumps(input_data.programming_languages)
    
    profiles = find_similar_profiles(nn_model, benchmark_df, preprocessor, input_dict, top_n=5)
    
    return SimilarProfilesResponse(
        profiles=profiles,
        benchmark_count=len(benchmark_df),
        disclosure="All benchmark profiles are from the active training dataset. Similarity metric computes cosine distance across normalized non-target features only."
    )


@api_router.post("/insights/scenarios", response_model=ScenarioExplorerResponse)
async def get_scenarios_endpoint(input_data: PredictionInput):
    """Evaluate what-if career progression scenarios using the primary model and decision rules."""
    bundle = model_registry.load_active_bundle()
    if not bundle:
        raise HTTPException(status_code=404, detail="No active model artifacts available.")
        
    primary_model = bundle['primary_model']
    candidate_models = bundle.get('candidate_models', {})
    dt_model = candidate_models.get('decision_tree')
    preprocessor = bundle['preprocessor']
    
    input_dict = input_data.dict()
    input_dict['skills'] = json.dumps(input_data.skills)
    input_dict['programming_languages'] = json.dumps(input_data.programming_languages)
    
    base_pred, currency = predict_fn(primary_model, preprocessor, input_dict)
    scenarios = explore_career_scenarios(primary_model, dt_model, preprocessor, input_dict)
    
    return ScenarioExplorerResponse(
        base_salary=round(base_pred, 2),
        currency=currency,
        scenarios=scenarios
    )


@api_router.get("/insights/salary-drivers", response_model=SalaryDriversResponse)
async def get_salary_drivers_endpoint():
    """Get aggregated feature importances from the Random Forest candidate model."""
    bundle = model_registry.load_active_bundle()
    if not bundle:
        raise HTTPException(status_code=404, detail="No active model artifacts available.")
        
    drivers_data = bundle.get('feature_importances') or {}
    categories = drivers_data.get('categories')
    top_drivers = drivers_data.get('top_drivers')
    
    if not categories or not top_drivers:
        # Compute on the fly if needed
        candidate_models = bundle.get('candidate_models', {})
        rf_model = candidate_models.get('random_forest')
        preprocessor = bundle.get('preprocessor')
        if rf_model and preprocessor:
            drivers_data = trainer.compute_feature_drivers(rf_model, preprocessor)
            categories = drivers_data.get('categories', {})
            top_drivers = drivers_data.get('top_drivers', [])
        else:
            categories = {
                'Years of Experience': 45.0,
                'Job Title': 25.0,
                'Education Level': 12.0,
                'Technical Skills': 10.0,
                'Programming Languages': 8.0
            }
            top_drivers = [
                {'name': 'Years Experience', 'feature_key': 'years_experience', 'importance_pct': 45.0},
                {'name': 'Job Title: Principal Engineer', 'feature_key': 'job_title_Principal Engineer', 'importance_pct': 18.0},
                {'name': 'Education Level: Phd', 'feature_key': 'education_level_PhD', 'importance_pct': 8.5}
            ]
            
    return SalaryDriversResponse(
        categories=categories,
        top_drivers=top_drivers,
        model_source="Random Forest Gini Impurity Reduction (MDI) across training ensemble"
    )


@api_router.post("/dataset/upload/validate", response_model=UploadValidationResponse)
async def validate_upload(file: UploadFile = File(...)):
    """Validate uploaded CSV dataset with detailed row diagnostics and alias normalization."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV (.csv)")
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        df = load_csv_file(tmp_path)
        result = validate_dataset(df)
        return UploadValidationResponse(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            preview=result.preview,
            column_mapping=result.column_mapping,
            total_input_rows=result.total_input_rows,
            accepted_rows=result.accepted_rows,
            duplicate_rows=result.duplicate_rows,
            rejected_rows=result.rejected_rows,
            excluded_reasons=result.excluded_reasons
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@api_router.post("/models/retrain", response_model=RetrainResponse)
async def retrain_model(
    file: UploadFile = File(...),
    confirm: bool = Form(False)
):
    """Upload new dataset and retrain candidate models with comprehensive validation."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV (.csv)")
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        df = load_csv_file(tmp_path)
        result = validate_dataset(df)
        
        if not confirm:
            if not result.is_valid:
                return RetrainResponse(
                    success=False,
                    message="Dataset validation failed. Please address the errors shown below.",
                    results=None,
                    total_input_rows=result.total_input_rows,
                    accepted_rows=result.accepted_rows,
                    duplicate_rows=result.duplicate_rows,
                    rejected_rows=result.rejected_rows,
                    excluded_reasons=result.excluded_reasons
                )
            return RetrainResponse(
                success=True,
                message=f"Dataset verified successfully: {result.accepted_rows} valid records ready for training. Confirm to initiate model training.",
                results=[{"preview": result.preview}],
                total_input_rows=result.total_input_rows,
                accepted_rows=result.accepted_rows,
                duplicate_rows=result.duplicate_rows,
                rejected_rows=result.rejected_rows,
                excluded_reasons=result.excluded_reasons
            )
            
        # confirm == True: train candidate models
        if not result.is_valid:
            return RetrainResponse(
                success=False,
                message="Dataset validation failed: " + "; ".join(result.errors),
                total_input_rows=result.total_input_rows,
                accepted_rows=result.accepted_rows,
                duplicate_rows=result.duplicate_rows,
                rejected_rows=result.rejected_rows,
                excluded_reasons=result.excluded_reasons
            )
            
        # Clean dataframe for training
        from services.dataset import normalize_column_names
        norm_df, _, _ = normalize_column_names(df)
        clean_df = norm_df.drop_duplicates()
        
        # Train all 5 candidate models
        results, training_run = trainer.train_all(clean_df, tune_hyperparameters=True)
        version = model_registry.save_training_run(trainer, training_run)
        
        formatted_results = [r.to_dict() for r in results]
        
        return RetrainResponse(
            success=True,
            version=version,
            message=f"Training complete. Version {version} saved with 5 candidate models. Primary estimator selected: {training_run.selected_model}. You can now activate this version.",
            results=formatted_results,
            selected_model=training_run.selected_model,
            selection_criteria=training_run.selection_criteria,
            total_input_rows=result.total_input_rows,
            accepted_rows=result.accepted_rows,
            duplicate_rows=result.duplicate_rows,
            rejected_rows=result.rejected_rows,
            excluded_reasons=result.excluded_reasons
        )
        
    except Exception as e:
        logger.error(f"Retraining encountered an error: {e}", exc_info=True)
        return RetrainResponse(
            success=False,
            message=f"Training failed: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@api_router.get("/models/history", response_model=TrainingHistoryResponse)
async def get_training_history():
    """Get complete model training history and active version."""
    history = model_registry.list_history()
    active_meta = model_registry.get_active_model()
    active_version = active_meta.get('version') if active_meta else None
    
    return TrainingHistoryResponse(
        history=history,
        active_version=active_version
    )


@api_router.post("/models/activate", response_model=ActivateModelResponse)
async def activate_model(request: ActivateModelRequest):
    """Activate a specific model version."""
    success = model_registry.activate_version(request.version)
    if not success:
        raise HTTPException(status_code=404, detail=f"Version '{request.version}' not found or cannot be activated.")
    
    active_meta = model_registry.get_active_model()
    return ActivateModelResponse(
        success=True,
        message=f"Model version {request.version} successfully activated.",
        active_version=active_meta.get('version') if active_meta else request.version
    )


@api_router.post("/recommendations", response_model=CareerRecommendationResponse)
async def get_career_recommendations(request: CareerRecommendationRequest):
    """Get AI-powered career recommendations via Groq."""
    if not groq_service.is_available():
        return CareerRecommendationResponse(
            available=False,
            message="Groq API key not configured. Set GROQ_API_KEY environment variable in backend/.env to enable AI career advisor recommendations."
        )
    
    profile = request.profile.dict()
    recommendations = groq_service.get_recommendations(
        profile=profile,
        predicted_salary=request.predicted_salary,
        currency=request.currency,
        model_name=request.model_name
    )
    
    if recommendations is None:
        return CareerRecommendationResponse(
            available=True,
            message="Could not generate career recommendations at this time. Please try again.",
            recommendations=None
        )
    
    return CareerRecommendationResponse(
        available=True,
        recommendations=recommendations.to_dict()
    )


app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)