import React, { useState, useEffect } from 'react';
import { api } from '../utils/api';

function CareerRecommendations({ profile, predictedSalary, currency, modelName, onClose }) {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [available, setAvailable] = useState(true);

  useEffect(() => {
    fetchRecommendations();
  }, [profile, predictedSalary, currency, modelName]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getCareerRecommendations({
        profile,
        predicted_salary: predictedSalary,
        currency,
        model_name: modelName
      });
      
      if (!response.available) {
        setAvailable(false);
        setError(response.message);
        setRecommendations(null);
      } else {
        setAvailable(true);
        setRecommendations(response.recommendations);
      }
    } catch (err) {
      setError(err.message);
      setRecommendations(null);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="recommendation-section">
        <div className="loading">
          <div className="spinner"></div>
          Generating AI career recommendations...
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onClose} style={{ marginTop: '16px' }}>
          Close
        </button>
      </div>
    );
  }

  if (!available) {
    return (
      <div className="recommendation-section">
        <div className="alert alert-warning">
          <strong>AI Recommendations Unavailable</strong>
          <p style={{ marginTop: '8px' }}>{error || 'Groq API not configured.'}</p>
          <p style={{ marginTop: '8px', fontSize: '0.9rem' }}>
            To enable: Set GROQ_API_KEY in backend/.env and restart the backend server.
            Get a key from <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer">console.groq.com</a>
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onClose}>
          Close
        </button>
      </div>
    );
  }

  if (error) {
    return (
      <div className="recommendation-section">
        <div className="alert alert-error">
          <strong>Error:</strong> {error}
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onClose}>
          Close
        </button>
      </div>
    );
  }

  if (!recommendations) {
    return (
      <div className="recommendation-section">
        <div className="alert alert-error">No recommendations received.</div>
        <button className="btn btn-secondary btn-sm" onClick={onClose}>Close</button>
      </div>
    );
  }

  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: currency, minimumFractionDigits: 0 }).format(val);
  };

  return (
    <div className="recommendation-section">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3>AI Career Recommendations</h3>
        <button className="btn btn-secondary btn-sm" onClick={onClose}>Close</button>
      </div>
      
      <div className="alert alert-info" style={{ fontSize: '0.85rem' }}>
        <strong>Based on:</strong> Your profile + Predicted salary of {formatCurrency(predictedSalary)} using {modelName}.
        <br/>
        <strong>Important:</strong> These are AI-generated suggestions for career development. The salary prediction is an estimate from a model trained on synthetic data. Not guaranteed outcomes.
      </div>

      <div className="recommendation-grid">
        {recommendations.skills_to_learn?.length > 0 && (
          <div className="recommendation-card">
            <h4>🎯 Skills to Learn</h4>
            <ul>
              {recommendations.skills_to_learn.map((skill, i) => (
                <li key={i}>{skill}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.learning_roadmap?.length > 0 && (
          <div className="recommendation-card">
            <h4>📚 Learning Roadmap</h4>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {recommendations.learning_roadmap.map((item, i) => (
                <li key={i} className="roadmap-item">
                  <span className={`roadmap-priority priority-${(item.priority || 'medium').toLowerCase()}`}>
                    {item.priority}
                  </span>
                  <div>
                    <strong>{item.topic}</strong>
                    <div style={{ fontSize: '0.8rem', color: '#7f8c8d' }}>{item.description}</div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.portfolio_projects?.length > 0 && (
          <div className="recommendation-card">
            <h4>💼 Portfolio Projects</h4>
            <ul>
              {recommendations.portfolio_projects.map((project, i) => (
                <li key={i}>{project}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.certifications?.length > 0 && (
          <div className="recommendation-card">
            <h4>📜 Certifications</h4>
            <ul>
              {recommendations.certifications.map((cert, i) => (
                <li key={i}>{cert}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.career_suggestions?.length > 0 && (
          <div className="recommendation-card">
            <h4>🚀 Career Suggestions</h4>
            <ul>
              {recommendations.career_suggestions.map((suggestion, i) => (
                <li key={i}>{suggestion}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.limitations?.length > 0 && (
          <div className="recommendation-card" style={{ background: '#fef9e7', border: '1px solid #f9e79f' }}>
            <h4>⚠️ Limitations & Caveats</h4>
            <ul>
              {recommendations.limitations.map((lim, i) => (
                <li key={i} style={{ fontSize: '0.8rem', color: '#b7950b' }}>{lim}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default CareerRecommendations;