import React, { useState, useCallback } from 'react';
import { api } from '../utils/api';

function ModelManagement({ trainingHistory, onModelActivated, onRetrainComplete }) {
  const [activeStep, setActiveStep] = useState('history'); // 'history', 'upload', 'validating', 'training', 'results'
  const [uploadFile, setUploadFile] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        setError('Please select a CSV file');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }
      setUploadFile(file);
      setError(null);
      setValidationResult(null);
      setActiveStep('validating');
      validateFile(file);
    }
  };

  const validateFile = async (file) => {
    try {
      const result = await api.validateDataset(file);
      setValidationResult(result);
      if (result.is_valid) {
        setActiveStep('upload'); // Show confirm screen
      } else {
        setActiveStep('validating'); // Stay on validation with errors
      }
    } catch (err) {
      setError(err.message);
      setActiveStep('upload');
    }
  };

  const handleConfirmTraining = async () => {
    if (!uploadFile) return;
    setTrainingProgress(true);
    setError(null);
    setActiveStep('training');
    
    try {
      const result = await api.retrainModel(uploadFile, true);
      setTrainingResult(result);
      if (result.success) {
        setActiveStep('results');
      } else {
        setActiveStep('upload');
        setError(result.message);
      }
    } catch (err) {
      setError(err.message);
      setActiveStep('upload');
    } finally {
      setTrainingProgress(false);
    }
  };

  const handleActivate = async (version) => {
    if (!window.confirm(`Activate model version ${version}? This will replace the current active model.`)) {
      return;
    }
    try {
      await api.activateModel(version);
      onModelActivated();
      setTrainingHistory(prev => prev.map(h => ({ ...h, is_active: h.version === version })));
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRetry = () => {
    setUploadFile(null);
    setValidationResult(null);
    setTrainingResult(null);
    setError(null);
    setActiveStep('upload');
  };

  const formatDate = (isoString) => {
    try {
      return new Date(isoString).toLocaleString();
    } catch {
      return isoString;
    }
  };

  const formatCurrency = (val) => {
    if (val === null || val === undefined) return 'N/A';
    if (val >= 1e6) return `$${(val/1e6).toFixed(2)}M`;
    if (val >= 1e3) return `$${(val/1e3).toFixed(1)}K`;
    return `$${val.toFixed(0)}`;
  };

  return (
    <div>
      <div className="card">
        <h2>Model Management</h2>
        <p style={{ marginBottom: '20px', color: '#7f8c8d' }}>
          Upload a new CSV dataset to retrain all 5 models. The new model version will be saved 
          but not activated until you review results and confirm.
        </p>

        {error && <div className="alert alert-error" role="alert">{error}</div>}

        {/* Step 1: Upload */}
        {(activeStep === 'upload' || activeStep === 'history') && (
          <div>
            <label className="upload-area" onClick={() => document.getElementById('csv-upload').click()}>
              <input
                id="csv-upload"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              <div className="upload-icon">📁</div>
              <div className="upload-text">Click or drag to upload a CSV dataset</div>
              <div className="upload-hint">Max 10MB • Must contain required columns</div>
            </label>

            <div style={{ marginTop: '16px', padding: '16px', background: '#f8f9fa', borderRadius: '8px', fontSize: '0.85rem' }}>
              <strong>Required CSV Columns:</strong>
              <ul style={{ marginLeft: '20px', marginTop: '8px', lineHeight: '1.8' }}>
                <li><code>years_experience</code> (numeric, ≥0)</li>
                <li><code>education_level</code> (categorical)</li>
                <li><code>job_title</code> (categorical)</li>
                <li><code>skills</code> (JSON array, e.g., <code>["Git","Docker"]</code>)</li>
                <li><code>programming_languages</code> (JSON array, e.g., <code>["Python","JavaScript"]</code>)</li>
                <li><code>country</code> (categorical)</li>
                <li><code>annual_salary_usd</code> (numeric, greater than 0)</li>
              </ul>
            </div>
          </div>
        )}

        {/* Step 2: Validation Results */}
        {activeStep === 'validating' && validationResult && (
          <div>
            <h3>Dataset Validation Results</h3>
            <div className={`validation-result ${validationResult.is_valid ? 'validation-success' : 'validation-error'}`}>
              <h4>{validationResult.is_valid ? '✅ Valid Dataset' : '❌ Validation Failed'}</h4>
              
              {validationResult.errors.length > 0 && (
                <div>
                  <strong>Errors:</strong>
                  <ul>{validationResult.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
                </div>
              )}
              
              {validationResult.warnings.length > 0 && (
                <div style={{ marginTop: '12px' }}>
                  <strong>Warnings:</strong>
                  <ul>{validationResult.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                </div>
              )}

              <details style={{ marginTop: '16px' }}>
                <summary>Preview (first 5 rows)</summary>
                <pre style={{ marginTop: '8px', fontSize: '0.75rem', overflow: 'auto', maxHeight: '200px' }}>
                  {JSON.stringify(validationResult.preview.head, null, 2)}
                </pre>
              </details>
            </div>

            {validationResult.is_valid && (
              <div style={{ marginTop: '20px', display: 'flex', gap: '12px' }}>
                <button className="btn btn-primary" onClick={handleConfirmTraining}>
                  Confirm & Train Models
                </button>
                <button className="btn btn-secondary" onClick={handleRetry}>
                  Choose Different File
                </button>
              </div>
            )}

            {!validationResult.is_valid && (
              <button className="btn btn-secondary" onClick={handleRetry} style={{ marginTop: '16px' }}>
                Choose Different File
              </button>
            )}
          </div>
        )}

        {/* Step 3: Training Progress */}
        {activeStep === 'training' && (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <div className="spinner" style={{ margin: '0 auto 16px' }}></div>
            <h3>Training Models...</h3>
            <p style={{ color: '#7f8c8d' }}>Training 5 algorithms with hyperparameter tuning. This may take 30-60 seconds.</p>
            <div style={{ marginTop: '16px', fontSize: '0.9rem', color: '#3498db' }}>
              Please wait...
            </div>
          </div>
        )}

        {/* Step 4: Training Results */}
        {activeStep === 'results' && trainingResult && (
          <div>
            <div className={`validation-result ${trainingResult.success ? 'validation-success' : 'validation-error'}`}>
              <h4>{trainingResult.success ? '✅ Training Complete' : '❌ Training Failed'}</h4>
              <p>{trainingResult.message}</p>
              
              {trainingResult.version && (
                <p><strong>Version:</strong> {trainingResult.version}</p>
              )}
              
              {trainingResult.selected_model && (
                <p><strong>Selected Model:</strong> {trainingResult.results?.find(r => r.algorithm === trainingResult.selected_model)?.model_name}</p>
              )}
              
              {trainingResult.selection_criteria && (
                <p><strong>Selection:</strong> {trainingResult.selection_criteria}</p>
              )}
            </div>

            {trainingResult.results && trainingResult.results.length > 0 && (
              <div style={{ marginTop: '20px' }}>
                <h3>Model Results</h3>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Val MAE</th>
                        <th>Test MAE</th>
                        <th>Val R²</th>
                        <th>Test R²</th>
                        <th>Time (s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingResult.results.map((r, i) => (
                        <tr key={i} style={{ fontWeight: r.algorithm === trainingResult.selected_model ? '600' : 'normal' }}>
                          <td>
                            {r.model_name}
                            {r.algorithm === trainingResult.selected_model && (
                              <span className="badge badge-selected" style={{ marginLeft: '6px', fontSize: '0.65rem' }}>SELECTED</span>
                            )}
                          </td>
                          <td>{formatCurrency(r.validation_metrics?.mae)}</td>
                          <td>{formatCurrency(r.test_metrics?.mae)}</td>
                          <td>{r.validation_metrics?.r2?.toFixed(4)}</td>
                          <td>{r.test_metrics?.r2?.toFixed(4)}</td>
                          <td>{r.training_time_seconds?.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <div style={{ marginTop: '20px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
              {trainingResult.success && trainingResult.version && (
                <button 
                  className="btn btn-success" 
                  onClick={() => handleActivate(trainingResult.version)}
                >
                  Activate This Model
                </button>
              )}
              <button className="btn btn-secondary" onClick={handleRetry}>
                Upload Another Dataset
              </button>
              <button className="btn btn-primary" onClick={() => { onRetrainComplete(); setActiveStep('history'); }}>
                View History
              </button>
            </div>
          </div>
        )}

        {/* Training History */}
        {(activeStep === 'history' || activeStep === 'upload') && trainingHistory.length > 0 && (
          <div style={{ marginTop: '32px' }}>
            <h3>Training History</h3>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Dataset</th>
                    <th>Rows</th>
                    <th>Selected Model</th>
                    <th>Val MAE</th>
                    <th>Test MAE</th>
                    <th>Date</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingHistory.map((run, i) => (
                    <tr key={i}>
                      <td><code>{run.version}</code></td>
                      <td>{run.dataset_name}</td>
                      <td>{run.dataset_rows?.toLocaleString()}</td>
                      <td>
                        {run.results?.find(r => r.algorithm === run.selected_model)?.model_name || run.selected_model}
                      </td>
                      <td>{formatCurrency(run.results?.find(r => r.algorithm === run.selected_model)?.validation_metrics?.mae)}</td>
                      <td>{formatCurrency(run.results?.find(r => r.algorithm === run.selected_model)?.test_metrics?.mae)}</td>
                      <td>{formatDate(run.timestamp)}</td>
                      <td>
                        {run.is_active ? (
                          <span className="badge badge-active">ACTIVE</span>
                        ) : (
                          <span className="badge badge-inactive">Inactive</span>
                        )}
                      </td>
                      <td>
                        {!run.is_active && (
                          <button 
                            className="btn btn-sm btn-primary"
                            onClick={() => handleActivate(run.version)}
                          >
                            Activate
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {trainingHistory.length === 0 && activeStep === 'history' && (
          <div className="alert alert-info" style={{ marginTop: '20px' }}>
            No training history yet. Upload a dataset to train your first model.
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelManagement;