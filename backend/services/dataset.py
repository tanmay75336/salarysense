import pandas as pd
import numpy as np
import hashlib
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import ast
import os


@dataclass
class ValidationResult:
    """Result of dataset validation with row-level diagnostics."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    preview: Dict[str, Any]
    column_mapping: Dict[str, str]
    row_count: int
    column_count: int
    total_input_rows: int = 0
    accepted_rows: int = 0
    duplicate_rows: int = 0
    rejected_rows: int = 0
    excluded_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'preview': self.preview,
            'column_mapping': self.column_mapping,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'total_input_rows': self.total_input_rows,
            'accepted_rows': self.accepted_rows,
            'duplicate_rows': self.duplicate_rows,
            'rejected_rows': self.rejected_rows,
            'excluded_reasons': self.excluded_reasons or {}
        }


REQUIRED_COLUMNS = [
    'years_experience',
    'education_level',
    'job_title',
    'skills',
    'programming_languages',
    'country',
    'annual_salary_usd'
]

COLUMN_ALIASES = {
    'years_experience': ['experience_years', 'experience', 'yoe', 'years_of_experience', 'total_experience'],
    'education_level': ['education', 'degree', 'qualification', 'degree_level'],
    'job_title': ['role', 'title', 'position', 'job_role', 'designation'],
    'skills': ['skill_set', 'technical_skills', 'skills_list'],
    'programming_languages': ['languages', 'coding_languages', 'programming_language'],
    'country': ['location', 'nation', 'region'],
    'annual_salary_usd': ['salary', 'annual_salary', 'salary_usd', 'compensation', 'target_salary', 'salary_in_usd']
}

CATEGORICAL_COLUMNS = ['education_level', 'job_title', 'country']
NUMERIC_COLUMNS = ['years_experience']
MULTILABEL_COLUMNS = ['skills', 'programming_languages']
TARGET_COLUMN = 'annual_salary_usd'

EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD']
KNOWN_COUNTRIES = ['United States', 'India', 'Germany', 'United Kingdom', 'Canada', 'Australia']
MIN_ROWS = 10
MAX_FILE_SIZE_MB = 10


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Standardize DataFrame column names using exact names or recognized aliases.
    Returns (normalized_df, column_mapping, missing_columns).
    """
    df = df.copy()
    cleaned_col_map = {str(col).strip().lower(): col for col in df.columns}
    
    mapping = {}
    rename_dict = {}
    
    for req in REQUIRED_COLUMNS:
        req_clean = req.lower()
        if req in df.columns:
            mapping[req] = req
        elif req_clean in cleaned_col_map:
            orig = cleaned_col_map[req_clean]
            mapping[req] = orig
            rename_dict[orig] = req
        else:
            found = False
            for alias in COLUMN_ALIASES.get(req, []):
                alias_clean = alias.lower()
                if alias_clean in cleaned_col_map:
                    orig = cleaned_col_map[alias_clean]
                    mapping[req] = orig
                    rename_dict[orig] = req
                    found = True
                    break
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return df, mapping, missing


def parse_multilabel_item(val: Any) -> Optional[List[str]]:
    """Parse JSON list, Python literal list, or comma-delimited string."""
    if pd.isna(val) or val is None or val == '' or val == '[]':
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
        
    val_str = str(val).strip()
    if not val_str or val_str == '[]':
        return []
        
    try:
        parsed = json.loads(val_str)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return None
    except Exception:
        pass
        
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return None
    except Exception:
        pass
        
    # Comma-separated fallback only if string contains commas
    if ',' in val_str:
        cleaned = val_str.strip('[](){}"\' ')
        parts = [s.strip().strip('\'"') for s in cleaned.split(',') if s.strip()]
        if parts:
            return parts
            
    return None


def validate_csv_structure(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, str], pd.DataFrame]:
    """Validate CSV structure, perform alias mapping, and return errors/warnings."""
    errors = []
    warnings = []
    
    norm_df, mapping, missing = normalize_column_names(df)
    
    if missing:
        errors.append(f"Missing required columns: {missing}")
        
    extra = [c for c in norm_df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"Extra columns will be ignored during model training: {extra}")
        
    return errors, warnings, mapping, norm_df


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """
    Complete dataset validation with vectorized checks, alias normalization,
    and detailed row diagnostics.
    """
    total_input_rows = len(df)
    struct_errors, struct_warnings, column_mapping, norm_df = validate_csv_structure(df)
    
    errors = list(struct_errors)
    warnings = list(struct_warnings)
    excluded_reasons: Dict[str, int] = {
        'missing_required_fields': 0,
        'negative_or_invalid_experience': 0,
        'non_positive_or_invalid_salary': 0,
        'unparseable_multilabel': 0
    }
    
    if errors:
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            preview=generate_preview(norm_df),
            column_mapping=column_mapping,
            row_count=total_input_rows,
            column_count=len(norm_df.columns),
            total_input_rows=total_input_rows,
            accepted_rows=0,
            duplicate_rows=0,
            rejected_rows=total_input_rows,
            excluded_reasons=excluded_reasons
        )
    
    # 1. Duplicates
    dup_mask = norm_df.duplicated()
    duplicate_rows = int(dup_mask.sum())
    if duplicate_rows > 0:
        warnings.append(f"Found {duplicate_rows} exact duplicate rows. They will be deduplicated for training.")
    
    dedup_df = norm_df[~dup_mask].copy()
    
    # 2. Vectorized validation on required fields
    invalid_row_mask = pd.Series(False, index=dedup_df.index)
    
    # Check numeric experience
    exp_series = pd.to_numeric(dedup_df['years_experience'], errors='coerce')
    bad_exp_mask = exp_series.isna() | (exp_series < 0)
    if bad_exp_mask.any():
        bad_count = int(bad_exp_mask.sum())
        excluded_reasons['negative_or_invalid_experience'] = bad_count
        invalid_row_mask |= bad_exp_mask
        errors.append(f"Column 'years_experience' has {bad_count} invalid or negative values")
        
    high_exp = (exp_series > 50).sum()
    if high_exp > 0:
        warnings.append(f"Column 'years_experience' has {int(high_exp)} values > 50 years")
        
    # Check target annual_salary_usd
    salary_series = pd.to_numeric(dedup_df['annual_salary_usd'], errors='coerce')
    bad_sal_mask = salary_series.isna() | (salary_series <= 0)
    if bad_sal_mask.any():
        bad_count = int(bad_sal_mask.sum())
        excluded_reasons['non_positive_or_invalid_salary'] = bad_count
        invalid_row_mask |= bad_sal_mask
        errors.append(f"Target column 'annual_salary_usd' has {bad_count} non-positive or invalid numeric values")
        
    # Outlier warning on target
    valid_salaries = salary_series[~salary_series.isna() & (salary_series > 0)]
    if len(valid_salaries) >= 10:
        q1 = valid_salaries.quantile(0.25)
        q3 = valid_salaries.quantile(0.75)
        iqr = q3 - q1
        outliers = ((valid_salaries < q1 - 1.5 * iqr) | (valid_salaries > q3 + 1.5 * iqr)).sum()
        if outliers > 0:
            warnings.append(f"Target column has {int(outliers)} potential outliers (IQR method)")
            
    # Check categorical columns
    for col in CATEGORICAL_COLUMNS:
        empty_cat = dedup_df[col].isna() | (dedup_df[col].astype(str).str.strip() == '')
        if empty_cat.any():
            cnt = int(empty_cat.sum())
            excluded_reasons['missing_required_fields'] = excluded_reasons.get('missing_required_fields', 0) + cnt
            invalid_row_mask |= empty_cat
            errors.append(f"Column '{col}' has {cnt} missing or blank entries")
            
    # Check multilabel columns
    for col in MULTILABEL_COLUMNS:
        unparseable = 0
        for idx, val in dedup_df[col].items():
            parsed = parse_multilabel_item(val)
            if parsed is None:
                unparseable += 1
                invalid_row_mask.loc[idx] = True
        if unparseable > 0:
            excluded_reasons['unparseable_multilabel'] = excluded_reasons.get('unparseable_multilabel', 0) + unparseable
            errors.append(f"Column '{col}' has {unparseable} invalid JSON array values")
            
    rejected_rows = int(invalid_row_mask.sum())
    accepted_rows = len(dedup_df) - rejected_rows
    
    if accepted_rows < MIN_ROWS:
        errors.append(f"Dataset has only {accepted_rows} valid records after filtering. Minimum {MIN_ROWS} required for training.")
        
    if 10 <= accepted_rows < 30:
        warnings.append(f"Dataset has {accepted_rows} records. A larger dataset (e.g. 50+ rows) improves model generalization.")
        
    preview = generate_preview(norm_df)
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        preview=preview,
        column_mapping=column_mapping,
        row_count=total_input_rows,
        column_count=len(norm_df.columns),
        total_input_rows=total_input_rows,
        accepted_rows=max(0, accepted_rows),
        duplicate_rows=duplicate_rows,
        rejected_rows=rejected_rows,
        excluded_reasons=excluded_reasons
    )


def generate_preview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary preview of the dataset."""
    preview = {
        'head': df.head(5).to_dict(orient='records'),
        'dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {str(col): int(df[col].isnull().sum()) for col in df.columns},
        'unique_counts': {str(col): int(df[col].nunique()) for col in df.columns},
        'numeric_summary': {}
    }
    
    # Numeric summaries
    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            valid_vals = numeric_series.dropna()
            if not valid_vals.empty:
                preview['numeric_summary'][col] = {
                    'mean': float(valid_vals.mean()),
                    'std': float(valid_vals.std()) if len(valid_vals) > 1 else 0.0,
                    'min': float(valid_vals.min()),
                    'max': float(valid_vals.max()),
                    'median': float(valid_vals.median())
                }
            else:
                preview['numeric_summary'][col] = {
                    'mean': None, 'std': None, 'min': None, 'max': None, 'median': None
                }
    
    return preview


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute hash of dataset for versioning."""
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:8]


def load_csv_file(file_path: str) -> pd.DataFrame:
    """Load CSV with configurable size check."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File size {file_size / (1024*1024):.1f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")
        
    return df