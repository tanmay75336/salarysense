import os
import json
import shutil
import glob
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import asdict
import joblib

from .training import TrainingRun, ModelTrainer


class ModelRegistry:
    """Manage model versions, training history, and activation."""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.models_dir = os.path.abspath(models_dir)
        self.active_model_path = os.path.join(self.models_dir, "active")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.active_model_path, exist_ok=True)
    
    def _get_version_dir(self, version: str) -> str:
        return os.path.join(self.models_dir, version)
    
    def _list_versions(self) -> List[str]:
        """List all version directories sorted by timestamp (newest first)."""
        dirs = glob.glob(os.path.join(self.models_dir, "v*"))
        versions = [os.path.basename(d) for d in dirs if os.path.isdir(d)]
        versions.sort(reverse=True)
        return versions
    
    def save_training_run(self, trainer: ModelTrainer, training_run: TrainingRun) -> str:
        """Save a complete training run as a new version."""
        version_dir = self._get_version_dir(training_run.version)
        os.makedirs(version_dir, exist_ok=True)
        
        base_path = os.path.join(version_dir, "model")
        trainer.save_model(base_path, training_run)
        
        return training_run.version
    
    def activate_version(self, version: str) -> bool:
        """Activate a specific version as the current model."""
        version_dir = self._get_version_dir(version)
        if not os.path.exists(version_dir):
            return False
        
        # Check required files
        req_files = ['model_model.joblib', 'model_preprocessor.joblib', 'model_metadata.json']
        for fname in req_files:
            if not os.path.exists(os.path.join(version_dir, fname)):
                return False
                
        # Copy model files to active directory
        copy_files = req_files + ['model_bundle.joblib']
        for fname in copy_files:
            src = os.path.join(version_dir, fname)
            dst = os.path.join(self.active_model_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        return True
    
    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Load the currently active model metadata."""
        metadata_path = os.path.join(self.active_model_path, "model_metadata.json")
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception:
            return None
    
    def load_active_artifacts(self) -> Optional[Tuple[Any, Any]]:
        """Load active model and preprocessor for prediction."""
        model_path = os.path.join(self.active_model_path, "model_model.joblib")
        preprocessor_path = os.path.join(self.active_model_path, "model_preprocessor.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            return None
        
        try:
            model = joblib.load(model_path)
            from .preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor.load(preprocessor_path)
            return model, preprocessor
        except Exception:
            return None

    def load_active_bundle(self) -> Optional[Dict[str, Any]]:
        """Load full active bundle (primary model, candidate models, NN index, preprocessor, and metadata)."""
        artifacts = self.load_active_artifacts()
        if not artifacts:
            return None
        primary_model, preprocessor = artifacts
        metadata = self.get_active_model()
        
        bundle_path = os.path.join(self.active_model_path, "model_bundle.joblib")
        bundle_data = {}
        if os.path.exists(bundle_path):
            try:
                bundle_data = joblib.load(bundle_path)
            except Exception:
                bundle_data = {}
                
        return {
            'primary_model': primary_model,
            'preprocessor': preprocessor,
            'metadata': metadata,
            'selected_algorithm': metadata.get('selected_model') if metadata else None,
            'candidate_models': bundle_data.get('candidate_models', {}),
            'nearest_neighbors': bundle_data.get('nearest_neighbors'),
            'feature_importances': bundle_data.get('feature_importances', {}),
            'benchmark_dataset': bundle_data.get('benchmark_dataset')
        }
    
    def list_history(self) -> List[Dict[str, Any]]:
        """Get training history for all versions."""
        history = []
        active_meta = self.get_active_model()
        active_ver = active_meta.get('version') if active_meta else None
        
        for version in self._list_versions():
            metadata_path = os.path.join(self._get_version_dir(version), "model_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    meta['version'] = version
                    meta['is_active'] = (version == active_ver)
                    history.append(meta)
                except Exception:
                    pass
        return history
    
    def get_version_details(self, version: str) -> Optional[Dict[str, Any]]:
        """Get full details for a specific version."""
        metadata_path = os.path.join(self._get_version_dir(version), "model_metadata.json")
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def delete_version(self, version: str) -> bool:
        """Delete a version (cannot delete active)."""
        active = self.get_active_model()
        if active and active.get('version') == version:
            return False
        
        version_dir = self._get_version_dir(version)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            return True
        return False
    
    def has_active_model(self) -> bool:
        """Check if there's an active model."""
        return os.path.exists(os.path.join(self.active_model_path, "model_metadata.json")) and \
               os.path.exists(os.path.join(self.active_model_path, "model_model.joblib")) and \
               os.path.exists(os.path.join(self.active_model_path, "model_preprocessor.joblib"))