import React, { useState } from 'react';
import { api } from '../utils/api';

function DatasetModels({ 
  modelComparison, 
  trainingHistory, 
  datasetInfo, 
  onModelActivated, 
  onRetrainComplete 
}) {
  const [activeTabSub, setActiveTabSub] = useState('comparison'); // 'comparison', 'upload', 'history', 'dataset'
  const [uploadFile, setUploadFile] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [validating, setValidating] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const handleFileDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer?.files || e.target?.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const processFile = async (file) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Selected file must be a CSV (.csv)');
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setError('File size exceeds the 25MB safety limit.');
      return;
    }

    setUploadFile(file);
    setError(null);
    setSuccessMsg(null);
    setValidationResult(null);
    setTrainingResult(null);
    setValidating(true);
    setActiveTabSub('upload');

    try {
      const result = await api.validateDataset(file);
      setValidationResult(result);
    } catch (err) {
      setError(err.message || 'Validation request failed.');
    } finally {
      setValidating(false);
    }
  };

  const handleConfirmTraining = async () => {
    if (!uploadFile) return;
    setTrainingProgress(true);
    setError(null);
    setSuccessMsg(null);

    try {
      const result = await api.retrainModel(uploadFile, true);
      setTrainingResult(result);
      if (result.success) {
        setSuccessMsg(result.message);
        if (onRetrainComplete) onRetrainComplete();
      } else {
        setError(result.message || 'Model retraining failed.');
      }
    } catch (err) {
      setError(err.message || 'Retraining encountered an error.');
    } finally {
      setTrainingProgress(false);
    }
  };

  const handleActivateVersion = async (version) => {
    try {
      const res = await api.activateModel(version);
      setSuccessMsg(res.message);
      if (onModelActivated) onModelActivated();
    } catch (err) {
      setError(err.message || 'Failed to activate version.');
    }
  };

  const formatCurrency = (val) => {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(val);
  };

  const formatDate = (isoStr) => {
    try {
      return new Date(isoStr).toLocaleString();
    } catch {
      return isoStr;
    }
  };

  const models = modelComparison?.models || [];
  const selectedAlgo = modelComparison?.selected_model;

  return (
    <div className="dataset-models-container">
      {/* Top Navigation Bar */}
      <div className="subtabs-nav" role="tablist">
        <button
          className={`subtab-btn ${activeTabSub === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTabSub('comparison')}
          role="tab"
          aria-selected={activeTabSub === 'comparison'}
        >
          Model Candidates ({models.length || 5})
        </button>
        <button
          className={`subtab-btn ${activeTabSub === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTabSub('upload')}
          role="tab"
          aria-selected={activeTabSub === 'upload'}
        >
          CSV Upload & Training
        </button>
        <button
          className={`subtab-btn ${activeTabSub === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTabSub('history')}
          role="tab"
          aria-selected={activeTabSub === 'history'}
        >
          Training History ({trainingHistory.length})
        </button>
        <button
          className={`subtab-btn ${activeTabSub === 'dataset' ? 'active' : ''}`}
          onClick={() => setActiveTabSub('dataset')}
          role="tab"
          aria-selected={activeTabSub === 'dataset'}
        >
          Benchmark Dataset Summary
        </button>
      </div>

      {error && <div className="alert alert-error" role="alert">{error}</div>}
      {successMsg && <div className="alert alert-success" role="status">{successMsg}</div>}

      {/* VIEW 1: Candidate Model Comparison */}
      {activeTabSub === 'comparison' && (
        <div className="card">
          <div className="card-header-flex">
            <div>
              <h2>Candidate Model Evaluation & Selection</h2>
              <p className="text-muted">
                All 5 candidate regressors are trained using identical preprocessed feature matrices and train/validation splits. 
                The official production estimator is selected strictly by validation MAE.
              </p>
            </div>
            {modelComparison && (
              <span className="badge badge-info">
                Trained: {formatDate(modelComparison.training_timestamp)}
              </span>
            )}
          </div>

          <div className="table-responsive" style={{ marginTop: '16px' }}>
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Algorithm & Candidate</th>
                  <th>Validation MAE (Selection)</th>
                  <th>Test MAE (Held-Out)</th>
                  <th>Validation R²</th>
                  <th>Test R²</th>
                  <th>Training Time</th>
                  <th>Role in Application</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, idx) => {
                  const isSelected = m.algorithm === selectedAlgo;
                  return (
                    <tr key={idx} className={isSelected ? 'selected-row' : ''}>
                      <td>
                        <strong>{m.model_name}</strong>
                        {isSelected && (
                          <span className="badge badge-selected" style={{ marginLeft: '8px' }}>
                            OFFICIAL ESTIMATOR
                          </span>
                        )}
                        <div className="text-muted" style={{ fontSize: '0.75rem' }}>
                          <code>{m.algorithm}</code>
                        </div>
                      </td>
                      <td className="metric-val highlight-metric">
                        {formatCurrency(m.validation_metrics?.mae)}
                      </td>
                      <td className="metric-val">
                        {formatCurrency(m.test_metrics?.mae)}
                      </td>
                      <td className="metric-val">
                        {m.validation_metrics?.r2 ? m.validation_metrics.r2.toFixed(3) : 'N/A'}
                      </td>
                      <td className="metric-val">
                        {m.test_metrics?.r2 ? m.test_metrics.r2.toFixed(3) : 'N/A'}
                      </td>
                      <td>
                        {m.training_time_seconds ? `${m.training_time_seconds.toFixed(2)}s` : '<1s'}
                      </td>
                      <td>
                        <small className="text-muted">
                          {m.algorithm === 'linear_regression' && 'Feature 4: Interpretable Linear Baseline'}
                          {m.algorithm === 'ridge_regression' && 'L2 Regularized Baseline Comparison'}
                          {m.algorithm === 'decision_tree' && 'Feature 3: Career Scenario Split Rules'}
                          {m.algorithm === 'random_forest' && 'Feature 5: Salary Driver Feature Importances'}
                          {m.algorithm === 'gradient_boosting' && 'Non-Linear Residual Optimization'}
                        </small>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {modelComparison?.selection_criteria && (
            <div className="selection-rationale-box">
              <strong>Official Selection Decision:</strong> {modelComparison.selection_criteria}
            </div>
          )}
        </div>
      )}

      {/* VIEW 2: CSV Upload & Retraining */}
      {activeTabSub === 'upload' && (
        <div className="card">
          <h2>Dataset Ingestion & Model Retraining</h2>
          <p className="text-muted">
            Upload custom CSV datasets from 10 to 50,000+ rows. Automatic column alias mapping resolves standard naming variants.
          </p>

          {/* Drag & Drop Zone */}
          <div 
            className="dropzone-area"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => document.getElementById('csv-file-input').click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter') document.getElementById('csv-file-input').click(); }}
          >
            <input 
              id="csv-file-input"
              type="file" 
              accept=".csv" 
              style={{ display: 'none' }}
              onChange={(e) => e.target.files && processFile(e.target.files[0])}
            />
            <div className="dropzone-icon">📁</div>
            <div className="dropzone-title">
              {uploadFile ? uploadFile.name : 'Click to select or drag and drop a CSV file'}
            </div>
            <div className="dropzone-hint">
              Supports standard CSV formatting • Automatic alias detection • Vectorized deduplication
            </div>
          </div>

          {/* Validating indicator */}
          {validating && (
            <div className="validation-loading-box">
              <div className="spinner"></div>
              <span>Validating columns, multi-label arrays, and numerical values...</span>
            </div>
          )}

          {/* Validation Diagnostics Summary */}
          {validationResult && !validating && (
            <div className={`validation-summary-card ${validationResult.is_valid ? 'summary-valid' : 'summary-invalid'}`}>
              <h3>
                {validationResult.is_valid ? '✅ Dataset Verification Passed' : '⚠️ Dataset Ingestion Issues Detected'}
              </h3>

              {/* Exact 4 Row Metric Counters */}
              <div className="row-counts-grid">
                <div className="count-card">
                  <span className="count-number">{validationResult.total_input_rows.toLocaleString()}</span>
                  <span className="count-label">Total Input Rows</span>
                </div>
                <div className="count-card accepted-count">
                  <span className="count-number">{validationResult.accepted_rows.toLocaleString()}</span>
                  <span className="count-label">Accepted Clean Rows</span>
                </div>
                <div className="count-card duplicate-count">
                  <span className="count-number">{validationResult.duplicate_rows.toLocaleString()}</span>
                  <span className="count-label">Duplicate Rows Removed</span>
                </div>
                <div className="count-card rejected-count">
                  <span className="count-number">{validationResult.rejected_rows.toLocaleString()}</span>
                  <span className="count-label">Rejected / Invalid Rows</span>
                </div>
              </div>

              {/* Exclusion Reasons Breakdown */}
              {validationResult.excluded_reasons && Object.keys(validationResult.excluded_reasons).length > 0 && (
                <div className="exclusion-breakdown-box">
                  <strong>Detailed Exclusion Breakdown:</strong>
                  <ul>
                    {Object.entries(validationResult.excluded_reasons).map(([reason, count]) => (
                      count > 0 && (
                        <li key={reason}>
                          <code>{reason}</code>: {count} affected row(s)
                        </li>
                      )
                    ))}
                  </ul>
                </div>
              )}

              {/* Errors & Warnings */}
              {validationResult.errors?.length > 0 && (
                <div className="validation-errors-list">
                  <strong>Validation Errors:</strong>
                  <ul>
                    {validationResult.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </div>
              )}

              {validationResult.warnings?.length > 0 && (
                <div className="validation-warnings-list">
                  <strong>Dataset Notes & Warnings:</strong>
                  <ul>
                    {validationResult.warnings.map((warn, i) => (
                      <li key={i}>{warn}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Column Mapping Preview */}
              {validationResult.column_mapping && (
                <div className="column-mapping-box">
                  <strong>Detected Column Mappings:</strong>
                  <div className="mapping-tags-container">
                    {Object.entries(validationResult.column_mapping).map(([standard, original]) => (
                      <span key={standard} className="mapping-tag">
                        {original} → <code>{standard}</code>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="validation-actions" style={{ marginTop: '20px' }}>
                {validationResult.is_valid && (
                  <button 
                    className="btn btn-primary btn-lg"
                    onClick={handleConfirmTraining}
                    disabled={trainingProgress}
                  >
                    {trainingProgress ? 'Training 5 Candidate Regressors...' : 'Confirm & Train 5 Candidate Models'}
                  </button>
                )}
                <button 
                  className="btn btn-secondary"
                  onClick={() => {
                    setUploadFile(null);
                    setValidationResult(null);
                    setTrainingResult(null);
                  }}
                  disabled={trainingProgress}
                >
                  Choose Different File
                </button>
              </div>
            </div>
          )}

          {/* Training in Progress Indicator */}
          {trainingProgress && (
            <div className="training-progress-banner">
              <div className="spinner"></div>
              <div>
                <strong>Model Training in Progress</strong>
                <p className="text-muted">
                  Fitting Multiple Linear Regression, Ridge, Decision Tree, Random Forest, and Gradient Boosting. 
                  Performing validation MAE model selection and fitting Nearest Neighbors index...
                </p>
              </div>
            </div>
          )}

          {/* Retrain Result Card */}
          {trainingResult && trainingResult.success && (
            <div className="card training-success-card" style={{ marginTop: '20px' }}>
              <h3>🎉 Training Completed Successfully!</h3>
              <p><strong>Generated Version:</strong> <code>{trainingResult.version}</code></p>
              <p><strong>Winning Estimator:</strong> {trainingResult.selected_model.toUpperCase()}</p>
              <p><strong>Selection Criteria:</strong> {trainingResult.selection_criteria}</p>
              
              <div style={{ marginTop: '16px', display: 'flex', gap: '12px' }}>
                <button 
                  className="btn btn-success"
                  onClick={() => handleActivateVersion(trainingResult.version)}
                >
                  Activate Version {trainingResult.version} Immediately
                </button>
                <button 
                  className="btn btn-secondary"
                  onClick={() => setActiveTabSub('comparison')}
                >
                  Inspect Candidate Comparison Table
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* VIEW 3: Training History & Version Activation */}
      {activeTabSub === 'history' && (
        <div className="card">
          <h2>Model Version History & Activation</h2>
          <p className="text-muted">
            Manage artifact versions. Activate any version to switch production inferences immediately without server downtime.
          </p>

          <div className="table-responsive" style={{ marginTop: '16px' }}>
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Dataset</th>
                  <th>Records</th>
                  <th>Selected Model</th>
                  <th>Validation MAE</th>
                  <th>Created Date</th>
                  <th>Lifecycle Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {trainingHistory.map((run, i) => {
                  const best = run.results?.find(r => r.algorithm === run.selected_model);
                  return (
                    <tr key={i} className={run.is_active ? 'selected-row' : ''}>
                      <td>
                        <code>{run.version}</code>
                      </td>
                      <td>{run.dataset_name}</td>
                      <td>{run.dataset_rows?.toLocaleString()}</td>
                      <td>
                        <strong>{best?.model_name || run.selected_model}</strong>
                      </td>
                      <td>{formatCurrency(best?.validation_metrics?.mae)}</td>
                      <td>{formatDate(run.timestamp)}</td>
                      <td>
                        {run.is_active ? (
                          <span className="badge badge-active">ACTIVE IN PRODUCTION</span>
                        ) : (
                          <span className="badge badge-inactive">Archived</span>
                        )}
                      </td>
                      <td>
                        {!run.is_active && (
                          <button 
                            className="btn btn-sm btn-primary"
                            onClick={() => handleActivateVersion(run.version)}
                          >
                            Activate
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* VIEW 4: Benchmark Dataset Summary */}
      {activeTabSub === 'dataset' && datasetInfo && (
        <div className="card">
          <h2>Active Benchmark Dataset Disclosure</h2>
          <p className="text-muted">
            Structural disclosure regarding the benchmark data currently utilized for model training and nearest neighbor similarity queries.
          </p>

          <div className="dataset-stats-grid" style={{ marginTop: '16px' }}>
            <div className="stat-box">
              <span className="stat-label">Total Records</span>
              <span className="stat-val">{datasetInfo.row_count.toLocaleString()}</span>
            </div>
            <div className="stat-box">
              <span className="stat-label">Total Features</span>
              <span className="stat-val">{datasetInfo.column_count} columns</span>
            </div>
            <div className="stat-box">
              <span className="stat-label">Dataset Source</span>
              <span className="stat-val">{datasetInfo.is_synthetic ? 'Synthetic Educational' : 'Empirical Industry'}</span>
            </div>
          </div>

          <div style={{ marginTop: '24px' }}>
            <h3>Dataset Schema & Columns</h3>
            <div className="mapping-tags-container" style={{ marginTop: '8px' }}>
              {datasetInfo.columns?.map(col => (
                <span key={col} className="driver-pill">
                  {col} ({datasetInfo.dtypes?.[col] || 'string'})
                </span>
              ))}
            </div>
          </div>

          <div style={{ marginTop: '24px' }}>
            <h3>Known Dataset Limitations</h3>
            <ul className="styled-list text-muted" style={{ marginTop: '8px' }}>
              {datasetInfo.limitations?.map((lim, i) => (
                <li key={i}>{lim}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default DatasetModels;
