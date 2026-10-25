import React, { useState, useEffect } from 'react';
import { api } from './utils/api';
import SalaryPredictor from './components/SalaryPredictor';
import CareerInsights from './components/CareerInsights';
import DatasetModels from './components/DatasetModels';

function App() {
  const [activeTab, setActiveTab] = useState('predictor'); // 'predictor', 'insights', 'models'
  const [health, setHealth] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Shared state between Predictor and Career Insights
  const [sharedProfile, setSharedProfile] = useState({
    years_experience: '3.5',
    education_level: 'Bachelor',
    job_title: 'Software Engineer',
    skills: ['Git', 'SQL', 'Docker'],
    programming_languages: ['Python', 'JavaScript'],
    country: 'United States'
  });
  const [sharedSalary, setSharedSalary] = useState(120000);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [healthRes, datasetRes, modelRes, historyRes] = await Promise.all([
        api.getHealth().catch(err => ({ status: 'unhealthy', active_model: false, groq_available: false })),
        api.getDatasetInfo().catch(() => null),
        api.getModelComparison().catch(() => null),
        api.getTrainingHistory().catch(() => ({ history: [] }))
      ]);
      setHealth(healthRes);
      setDatasetInfo(datasetRes);
      setModelComparison(modelRes);
      setTrainingHistory(historyRes.history || []);
    } catch (err) {
      setError('Cannot connect to backend services. Ensure FastAPI backend is running on http://localhost:8000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleModelActivated = async () => {
    const [modelRes, historyRes] = await Promise.all([
      api.getModelComparison().catch(() => null),
      api.getTrainingHistory().catch(() => ({ history: [] }))
    ]);
    setModelComparison(modelRes);
    setTrainingHistory(historyRes.history || []);
  };

  const handleRetrainComplete = async () => {
    await loadInitialData();
  };

  const handleNavigateToInsights = (profile, salary) => {
    if (profile) setSharedProfile(profile);
    if (salary) setSharedSalary(salary);
    setActiveTab('insights');
  };

  if (loading) {
    return (
      <div className="app-loading-screen">
        <div className="spinner"></div>
        <p>Initializing SalarySense AI Engine...</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="brand-lockup">
            <div className="brand-wordmark">
              <h1>SalarySense AI</h1>
              <p className="subtitle">Intelligent Salary Prediction &amp; Career Recommendation System</p>
            </div>
          </div>
          <div className="header-status-badges">
            <span className={`status-pill ${health?.active_model ? 'status-online' : 'status-offline'}`}>
              Model: {health?.active_model ? 'Active' : 'Initializing'}
            </span>
            <span className={`status-pill ${health?.groq_available ? 'status-online' : 'status-warning'}`}>
              Groq Advisor: {health?.groq_available ? 'Connected' : 'Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="container">
        {error && (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        )}

        {health && !health.active_model && (
          <div className="alert alert-warning" role="alert">
            <strong>System Notice:</strong> No active model detected. The system is training the initial 5 candidate models automatically.
          </div>
        )}

        {/* 3 Main Tabs */}
        <nav className="tabs" role="tablist" aria-label="Main navigation">
          <button
            className={`tab ${activeTab === 'predictor' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictor')}
            role="tab"
            aria-selected={activeTab === 'predictor'}
            id="tab-predictor"
            aria-controls="panel-predictor"
          >
            Salary Predictor
          </button>
          <button
            className={`tab ${activeTab === 'insights' ? 'active' : ''}`}
            onClick={() => setActiveTab('insights')}
            role="tab"
            aria-selected={activeTab === 'insights'}
            id="tab-insights"
            aria-controls="panel-insights"
          >
            Career Insights
          </button>
          <button
            className={`tab ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => setActiveTab('models')}
            role="tab"
            aria-selected={activeTab === 'models'}
            id="tab-models"
            aria-controls="panel-models"
          >
            Dataset &amp; Models
          </button>
        </nav>

        {/* TAB 1: Salary Predictor */}
        {activeTab === 'predictor' && (
          <div id="panel-predictor" role="tabpanel" aria-labelledby="tab-predictor">
            <SalaryPredictor
              modelComparison={modelComparison}
              onNavigateToCareerInsights={handleNavigateToInsights}
              lastProfile={sharedProfile}
              onProfileUpdate={setSharedProfile}
            />
          </div>
        )}

        {/* TAB 2: Career Insights */}
        {activeTab === 'insights' && (
          <div id="panel-insights" role="tabpanel" aria-labelledby="tab-insights">
            <CareerInsights
              profile={sharedProfile}
              predictedSalary={sharedSalary}
              currency="USD"
              modelName={modelComparison?.selected_model ? modelComparison.selected_model.toUpperCase() : 'Primary Estimator'}
            />
          </div>
        )}

        {/* TAB 3: Dataset & Models */}
        {activeTab === 'models' && (
          <div id="panel-models" role="tabpanel" aria-labelledby="tab-models">
            <DatasetModels
              modelComparison={modelComparison}
              trainingHistory={trainingHistory}
              datasetInfo={datasetInfo}
              onModelActivated={handleModelActivated}
              onRetrainComplete={handleRetrainComplete}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>SalarySense AI — Intelligent Salary Prediction & Career Recommendation System</p>
        <p className="footer-disclaimer">
          Estimates are generated by validated machine learning models trained on benchmark educational data. Not intended as guaranteed compensation offers.
        </p>
      </footer>
    </div>
  );
}

export default App;