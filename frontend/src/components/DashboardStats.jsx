import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function DashboardStats({ health, datasetInfo, modelComparison }) {
  if (!datasetInfo) {
    return <div className="loading">Loading dashboard...</div>;
  }

  const stats = [
    { label: 'Dataset Records', value: datasetInfo.row_count.toLocaleString() },
    { label: 'Input Features', value: datasetInfo.column_count - 1 }, // exclude target
    { label: 'Active Model', value: health?.active_model ? 'Yes' : 'No' },
    { label: 'Groq AI', value: health?.groq_available ? 'Configured' : 'Not Configured' }
  ];

  // Prepare chart data from model comparison
  const chartData = modelComparison?.models?.map(m => ({
    name: m.model_name.length > 20 ? m.model_name.substring(0, 20) + '...' : m.model_name,
    mae: m.validation_metrics?.mae || 0,
    rmse: m.validation_metrics?.rmse || 0,
    r2: m.validation_metrics?.r2 || 0,
    selected: m.algorithm === modelComparison?.selected_model
  })) || [];

  const formatCurrency = (val) => {
    if (val >= 1e6) return `$${(val/1e6).toFixed(1)}M`;
    if (val >= 1e3) return `$${(val/1e3).toFixed(1)}K`;
    return `$${val.toFixed(0)}`;
  };

  return (
    <div>
      <div className="grid-3">
        {stats.map((stat, i) => (
          <div key={i} className="stat-card">
            <div className="stat-value">{stat.value}</div>
            <div className="stat-label">{stat.label}</div>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Dataset Overview</h2>
          <div className="section">
            <p><strong>Source:</strong> {datasetInfo.source}</p>
            <p><strong>Type:</strong> {datasetInfo.is_synthetic ? 'Synthetic Educational Data' : 'Real-world Data'}</p>
            <p><strong>Rows:</strong> {datasetInfo.row_count} | <strong>Columns:</strong> {datasetInfo.column_count}</p>
            <p><strong>Target:</strong> Annual Salary (USD)</p>
          </div>
          <h3>Features</h3>
          <ul style={{ marginLeft: '20px', marginTop: '8px', lineHeight: '1.8' }}>
            <li>Years of Experience (numeric)</li>
            <li>Education Level (categorical: High School, Bachelor, Master, PhD)</li>
            <li>Job Title (categorical: 10+ roles)</li>
            <li>Skills (multi-label: 30+ technical skills)</li>
            <li>Programming Languages (multi-label: 20+ languages)</li>
            <li>Country (categorical: 6 countries)</li>
          </ul>
          <h3 style={{ marginTop: '16px' }}>Limitations</h3>
          <ul style={{ marginLeft: '20px', marginTop: '8px', lineHeight: '1.8', color: '#7f8c8d', fontSize: '0.9rem' }}>
            {datasetInfo.limitations.map((lim, i) => <li key={i}>{lim}</li>)}
          </ul>
        </div>

        <div className="card">
          <h2>Model Performance Comparison</h2>
          {chartData.length > 0 ? (
            <>
              <p style={{ marginBottom: '12px', fontSize: '0.9rem', color: '#7f8c8d' }}>
                Validation metrics (80% train, 20% test split). Lower MAE/RMSE is better. Higher R² is better.
                {modelComparison?.selected_model && (
                  <span className="badge badge-selected" style={{ marginLeft: '8px' }}>
                    Selected: {modelComparison.models.find(m => m.algorithm === modelComparison.selected_model)?.model_name}
                  </span>
                )}
              </p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" tickFormatter={formatCurrency} />
                    <YAxis dataKey="name" type="category" width={150} />
                    <Tooltip 
                      formatter={(value, name) => [formatCurrency(value), name]}
                      labelFormatter={(name) => name}
                    />
                    <Bar dataKey="mae" name="MAE" fill="#3498db" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{ marginTop: '16px', fontSize: '0.85rem', color: '#7f8c8d' }}>
                <strong>Selection Criteria:</strong> {modelComparison?.selection_criteria || 'N/A'}
              </div>
            </>
          ) : (
            <div className="alert alert-info">
              No trained models available. Train a model in the Model Management tab.
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h2>About This System</h2>
        <div className="grid-2">
          <div>
            <h3>How It Works</h3>
            <ol style={{ marginLeft: '20px', lineHeight: '1.8' }}>
              <li>Enter your professional profile (experience, education, role, skills, languages, country)</li>
              <li>The system uses a trained regression model to predict annual salary in USD</li>
              <li>Compare 5 algorithms: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting</li>
              <li>Upload your own CSV to retrain models with your data</li>
              <li>Get AI-powered career recommendations via Groq (optional)</li>
            </ol>
          </div>
          <div>
            <h3>Algorithms Explained</h3>
            <ul style={{ marginLeft: '20px', lineHeight: '1.8' }}>
              <li><strong>Linear Regression:</strong> Fits a straight-line relationship. Simple & interpretable.</li>
              <li><strong>Ridge Regression:</strong> Linear with L2 regularization. Reduces overfitting.</li>
              <li><strong>Decision Tree:</strong> Non-linear splits. Captures interactions but can overfit.</li>
              <li><strong>Random Forest:</strong> Ensemble of trees. Robust default choice.</li>
              <li><strong>Gradient Boosting:</strong> Sequential error correction. Often highest accuracy.</li>
            </ul>
          </div>
        </div>
        <div style={{ marginTop: '16px', padding: '16px', background: '#fef9e7', borderRadius: '8px', border: '1px solid #f9e79f' }}>
          <strong>Disclaimer:</strong> This system uses a synthetic educational dataset (~100 records). 
          Predictions demonstrate ML pipeline functionality and do not represent real-world salary accuracy. 
          Actual salaries vary by company, negotiation, benefits, equity, and market conditions.
        </div>
      </div>
    </div>
  );
}

export default DashboardStats;