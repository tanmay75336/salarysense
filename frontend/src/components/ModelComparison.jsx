import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

function ModelComparison({ modelComparison, datasetInfo }) {
  if (!modelComparison) {
    return (
      <div className="card">
        <div className="alert alert-info">
          <h3>No Models Trained</h3>
          <p>Train a model first using the Model Management tab, or wait for the default model to train on startup.</p>
        </div>
      </div>
    );
  }

  const models = modelComparison.models || [];
  const selectedAlgo = modelComparison.selected_model;

  const formatMetric = (val, type) => {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    if (type === 'currency') {
      if (val >= 1e6) return `$${(val/1e6).toFixed(2)}M`;
      if (val >= 1e3) return `$${(val/1e3).toFixed(1)}K`;
      return `$${val.toFixed(0)}`;
    }
    if (type === 'r2') return val.toFixed(4);
    return val.toFixed(2);
  };

  const getMetricClass = (val, type, allVals) => {
    if (val === null || val === undefined) return '';
    if (type === 'r2') {
      if (val < 0) return 'metric-bad';
      if (val < 0.5) return 'metric-warn';
      return 'metric-good';
    }
    // For MAE/RMSE, lower is better
    const validVals = allVals.filter(v => v !== null && v !== undefined);
    if (validVals.length === 0) return '';
    const min = Math.min(...validVals);
    const max = Math.max(...validVals);
    if (val === min) return 'metric-good';
    if (val === max) return 'metric-bad';
    return 'metric-warn';
  };

  return (
    <div>
      <div className="card">
        <h2>Model Comparison</h2>
        <p style={{ marginBottom: '16px', color: '#7f8c8d' }}>
          All 5 algorithms trained on the same 80/20 train-test split. 
          Validation metrics shown below (cross-validation on training set). 
          Test metrics are on the held-out 20% test set.
        </p>

        {/* Validation Metrics Chart */}
        <h3>Validation MAE Comparison</h3>
        <div className="chart-container" style={{ height: '300px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={models} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tickFormatter={v => formatMetric(v, 'currency')} />
              <YAxis dataKey="model_name" type="category" width={200} />
              <Tooltip 
                formatter={(value, name) => [formatMetric(value, 'currency'), name]}
                labelFormatter={(name) => name}
              />
              <Legend />
              <Bar dataKey="validation_metrics.mae" name="Validation MAE" fill="#3498db" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Validation vs Test Comparison */}
        <h3 style={{ marginTop: '24px' }}>Validation vs Test Metrics</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Val MAE</th>
                <th>Test MAE</th>
                <th>Val RMSE</th>
                <th>Test RMSE</th>
                <th>Val R²</th>
                <th>Test R²</th>
                <th>CV MAE (mean±std)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, i) => {
                const valMAE = model.validation_metrics?.mae;
                const testMAE = model.test_metrics?.mae;
                const valRMSE = model.validation_metrics?.rmse;
                const testRMSE = model.test_metrics?.rmse;
                const valR2 = model.validation_metrics?.r2;
                const testR2 = model.test_metrics?.r2;
                const cvScores = model.cv_scores;
                
                const isSelected = model.algorithm === selectedAlgo;
                
                return (
                  <tr key={i} style={{ fontWeight: isSelected ? '600' : 'normal' }}>
                    <td>
                      {model.model_name}
                      {isSelected && <span className="badge badge-selected" style={{ marginLeft: '8px', fontSize: '0.7rem' }}>SELECTED</span>}
                    </td>
                    <td className={getMetricClass(valMAE, 'mae', models.map(m => m.validation_metrics?.mae))}>
                      {formatMetric(valMAE, 'currency')}
                    </td>
                    <td className={getMetricClass(testMAE, 'mae', models.map(m => m.test_metrics?.mae))}>
                      {formatMetric(testMAE, 'currency')}
                    </td>
                    <td className={getMetricClass(valRMSE, 'mae', models.map(m => m.validation_metrics?.rmse))}>
                      {formatMetric(valRMSE, 'currency')}
                    </td>
                    <td className={getMetricClass(testRMSE, 'mae', models.map(m => m.test_metrics?.rmse))}>
                      {formatMetric(testRMSE, 'currency')}
                    </td>
                    <td className={getMetricClass(valR2, 'r2', models.map(m => m.validation_metrics?.r2))}>
                      {formatMetric(valR2, 'r2')}
                    </td>
                    <td className={getMetricClass(testR2, 'r2', models.map(m => m.test_metrics?.r2))}>
                      {formatMetric(testR2, 'r2')}
                    </td>
                    <td>
                      {cvScores?.mean_cv_mae ? (
                        `${formatMetric(cvScores.mean_cv_mae[0], 'currency')} ± ${formatMetric(cvScores.std_cv_mae[0], 'currency')}`
                      ) : 'N/A'}
                    </td>
                    <td>
                      {model.warnings?.length > 0 ? (
                        <span className="badge" style={{ background: '#fef9e7', color: '#b7950b' }}>
                          ⚠ Warnings
                        </span>
                      ) : (
                        <span className="badge badge-inactive">OK</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '16px', padding: '16px', background: '#f8f9fa', borderRadius: '8px' }}>
          <strong>Selection Criteria:</strong> {modelComparison.selection_criteria}
          <br/>
          <strong>Training Time:</strong> {modelComparison.training_timestamp}
          <br/>
          <strong>Dataset:</strong> {modelComparison.dataset_info?.rows} rows, {modelComparison.dataset_info?.name}
        </div>
      </div>

      {/* Algorithm Explanations */}
      <div className="card">
        <h2>Algorithm Explanations</h2>
        <div className="grid-2">
          <div>
            <h3>Multiple Linear Regression</h3>
            <p>Fits a linear equation: salary = β₀ + β₁×experience + β₂×education + ... Assumes linear relationships between features and target. Fast, interpretable, but may underfit complex patterns.</p>
            
            <h3>Ridge Regression</h3>
            <p>Linear regression with L2 regularization (α penalty on coefficient magnitudes). Reduces overfitting by shrinking coefficients. Good when features are correlated. α tuned via cross-validation.</p>
            
            <h3>Decision Tree Regressor</h3>
            <p>Recursively splits data into regions based on feature thresholds. Captures non-linear interactions naturally. Prone to overfitting; controlled by max_depth, min_samples_split.</p>
          </div>
          <div>
            <h3>Random Forest Regressor</h3>
            <p>Ensemble of decision trees trained on bootstrap samples with random feature subsets. Predictions averaged across trees. Robust, handles non-linearity well, good default choice.</p>
            
            <h3>Gradient Boosting Regressor</h3>
            <p>Sequential ensemble: each tree corrects residuals of previous ones. Often achieves highest accuracy. Controlled by n_estimators, learning_rate, max_depth. Slower to train.</p>
            
            <h3>Evaluation Metrics</h3>
            <ul style={{ marginLeft: '20px', lineHeight: '1.8' }}>
              <li><strong>MAE (Mean Absolute Error):</strong> Average absolute difference. Interpretable in salary units.</li>
              <li><strong>MSE (Mean Squared Error):</strong> Average squared difference. Penalizes large errors more.</li>
              <li><strong>RMSE (Root MSE):</strong> Square root of MSE. Same units as salary.</li>
              <li><strong>R² Score:</strong> Proportion of variance explained. 1 = perfect, 0 = baseline, negative = worse than baseline.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Dataset Limitations */}
      {datasetInfo && (
        <div className="card">
          <h2>Dataset Limitations (Affects All Models)</h2>
          <ul style={{ marginLeft: '20px', lineHeight: '1.8', color: '#7f8c8d' }}>
            {datasetInfo.limitations.map((lim, i) => <li key={i}>{lim}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ModelComparison;