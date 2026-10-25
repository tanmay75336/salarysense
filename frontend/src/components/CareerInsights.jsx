import React, { useState, useEffect } from 'react';
import { api } from '../utils/api';

function CareerInsights({ 
  profile, 
  predictedSalary, 
  currency = 'USD', 
  modelName = 'Production Regressor' 
}) {
  const [currentProfile, setCurrentProfile] = useState(profile || {
    years_experience: 3.5,
    education_level: 'Bachelor',
    job_title: 'Software Engineer',
    skills: ['Git', 'SQL', 'Docker'],
    programming_languages: ['Python', 'JavaScript'],
    country: 'United States'
  });
  const [currentSalary, setCurrentSalary] = useState(predictedSalary || 120000);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isAvailable, setIsAvailable] = useState(true);

  // Sync with prop changes if passed from predictor
  useEffect(() => {
    if (profile) setCurrentProfile(profile);
    if (predictedSalary) setCurrentSalary(predictedSalary);
  }, [profile, predictedSalary]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getCareerRecommendations({
        profile: {
          years_experience: parseFloat(currentProfile.years_experience) || 1,
          education_level: currentProfile.education_level || 'Bachelor',
          job_title: currentProfile.job_title || 'Software Engineer',
          skills: currentProfile.skills || [],
          programming_languages: currentProfile.programming_languages || [],
          country: currentProfile.country || 'United States'
        },
        predicted_salary: parseFloat(currentSalary) || 100000,
        currency: currency,
        model_name: modelName
      });

      if (!response.available) {
        setIsAvailable(false);
        setError(response.message || 'Groq AI integration is not configured.');
        setRecommendations(null);
      } else {
        setIsAvailable(true);
        setRecommendations(response.recommendations);
      }
    } catch (err) {
      setError(err.message);
      setRecommendations(null);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      maximumFractionDigits: 0
    }).format(val);
  };

  return (
    <div className="career-insights-container">
      {/* Advisor Header & Input Context */}
      <div className="card">
        <div className="card-header-flex">
          <div>
            <h2>AI Career Advisor & Upskilling Roadmap</h2>
            <p className="text-muted">
              Decoupled intelligent guidance powered by Groq LLaMA-3. Generates high-impact skills, prioritized learning tracks, and portfolio ideas tailored to your target compensation and role.
            </p>
          </div>
          <span className="badge badge-primary">Groq LLaMA-3.1 Powered</span>
        </div>

        <div className="advisor-context-box">
          <div className="context-item">
            <span className="context-label">Target Role:</span>
            <strong>{currentProfile.job_title}</strong>
          </div>
          <div className="context-item">
            <span className="context-label">Experience:</span>
            <strong>{currentProfile.years_experience} Years</strong>
          </div>
          <div className="context-item">
            <span className="context-label">Education:</span>
            <strong>{currentProfile.education_level}</strong>
          </div>
          <div className="context-item">
            <span className="context-label">Estimated Salary:</span>
            <strong>{formatCurrency(currentSalary)}</strong>
          </div>
        </div>

        <div style={{ marginTop: '20px' }}>
          <button 
            className="btn btn-primary btn-lg" 
            onClick={fetchRecommendations}
            disabled={loading}
          >
            {loading ? 'Consulting Career Advisor...' : 'Generate Personalized Career Recommendations'}
          </button>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="card text-center" style={{ padding: '40px' }}>
          <div className="spinner" style={{ margin: '0 auto 16px' }}></div>
          <h3>Analyzing Career Trajectory...</h3>
          <p className="text-muted">Synthesizing learning roadmap, portfolio initiatives, and industry certifications.</p>
        </div>
      )}

      {/* Unavailable / Error Message */}
      {!isAvailable && (
        <div className="card">
          <div className="alert alert-warning">
            <h4>Groq API Key Required</h4>
            <p style={{ marginTop: '8px' }}>
              {error || 'The GROQ_API_KEY environment variable is not configured.'}
            </p>
            <p style={{ marginTop: '8px', fontSize: '0.9rem' }}>
              To activate AI recommendations, ensure your API key is in <code>backend/.env</code> and restart the backend.
            </p>
          </div>
        </div>
      )}

      {error && isAvailable && (
        <div className="alert alert-error" role="alert">
          <strong>Advisor Error:</strong> {error}
        </div>
      )}

      {/* Structured Recommendation Content */}
      {recommendations && (
        <div className="recommendations-results-wrapper">
          <div className="alert alert-info" style={{ marginBottom: '20px' }}>
            <strong>Advisor Briefing:</strong> Customized for <strong>{currentProfile.job_title}</strong> ({currentProfile.years_experience} YOE, {currentProfile.education_level}) targeting {formatCurrency(currentSalary)} benchmark compensation.
          </div>

          <div className="recommendation-sections-grid">
            {/* 1. Skills to Learn */}
            {recommendations.skills_to_learn?.length > 0 && (
              <div className="card insight-card">
                <h3>🎯 High-Demand Skills to Acquire</h3>
                <p className="text-muted" style={{ fontSize: '0.85rem' }}>Core technologies that unlock higher tier roles in your specialty.</p>
                <div className="tags-container" style={{ marginTop: '12px' }}>
                  {recommendations.skills_to_learn.map((skill, idx) => (
                    <span key={idx} className="skill-tag">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* 2. Prioritized Learning Roadmap */}
            {recommendations.learning_roadmap?.length > 0 && (
              <div className="card insight-card roadmap-card">
                <h3>📚 Prioritized Learning Roadmap</h3>
                <div className="roadmap-list" style={{ marginTop: '16px' }}>
                  {recommendations.learning_roadmap.map((item, idx) => {
                    const priorityClass = `priority-${(item.priority || 'medium').toLowerCase()}`;
                    return (
                      <div key={idx} className="roadmap-step">
                        <span className={`priority-tag ${priorityClass}`}>
                          {item.priority || 'Medium'}
                        </span>
                        <div className="step-body">
                          <strong>{item.topic}</strong>
                          <p>{item.description}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 3. Portfolio Projects */}
            {recommendations.portfolio_projects?.length > 0 && (
              <div className="card insight-card">
                <h3>💼 Demonstrable Portfolio Projects</h3>
                <p className="text-muted" style={{ fontSize: '0.85rem' }}>Realistic proof-of-work systems that validate senior capabilities to hiring teams.</p>
                <ul className="styled-list" style={{ marginTop: '12px' }}>
                  {recommendations.portfolio_projects.map((proj, idx) => (
                    <li key={idx}>
                      <span>{proj}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 4. Certifications */}
            {recommendations.certifications?.length > 0 && (
              <div className="card insight-card">
                <h3>📜 Recognized Certifications</h3>
                <ul className="styled-list" style={{ marginTop: '12px' }}>
                  {recommendations.certifications.map((cert, idx) => (
                    <li key={idx}>
                      <span>{cert}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 5. Career Suggestions */}
            {recommendations.career_suggestions?.length > 0 && (
              <div className="card insight-card">
                <h3>🚀 Strategic Career Suggestions</h3>
                <ul className="styled-list" style={{ marginTop: '12px' }}>
                  {recommendations.career_suggestions.map((sugg, idx) => (
                    <li key={idx}>
                      <span>{sugg}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 6. Caveats & Limitations */}
            {recommendations.limitations?.length > 0 && (
              <div className="card insight-card caveats-card">
                <h3>⚠️ Methodological Caveats & Boundaries</h3>
                <ul className="styled-list text-muted" style={{ marginTop: '12px', fontSize: '0.85rem' }}>
                  {recommendations.limitations.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default CareerInsights;
