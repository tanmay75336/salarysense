import React, { useState, useEffect } from 'react';
import { api } from '../utils/api';
import MultiSelect from './MultiSelect';

const EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD'];
const JOB_TITLES = [
  'Junior Developer',
  'Software Engineer',
  'Senior Software Engineer',
  'Lead Engineer',
  'Principal Engineer',
  'Data Analyst',
  'Data Scientist',
  'Senior Data Scientist',
  'Principal Data Scientist',
  'DevOps Engineer',
  'Senior DevOps Engineer',
  'Principal DevOps Engineer'
];
const COUNTRIES = ['United States', 'India', 'Germany', 'United Kingdom', 'Canada', 'Australia'];
const ALL_SKILLS = [
  'Git', 'SQL', 'Docker', 'AWS', 'Kubernetes', 'CI/CD', 'Terraform',
  'Microservices', 'Architecture', 'Mentoring', 'Research', 'Strategy',
  'Excel', 'Tableau', 'PowerBI', 'ML', 'Deep Learning', 'MLOps',
  'Monitoring', 'Bash', 'Linux', 'Networking', 'Security', 'Testing',
  'Agile', 'Scrum', 'System Design', 'API Design', 'Databases', 'NoSQL'
];
const ALL_LANGUAGES = [
  'Python', 'JavaScript', 'TypeScript', 'Java', 'Go', 'Rust', 'C++',
  'Scala', 'Haskell', 'R', 'SQL', 'Bash', 'C#', 'PHP', 'Ruby',
  'Swift', 'Kotlin', 'Dart', 'Lua', 'Perl'
];

function SalaryPredictor({ 
  modelComparison, 
  onNavigateToCareerInsights, 
  lastProfile, 
  onProfileUpdate 
}) {
  const [formData, setFormData] = useState(lastProfile || {
    years_experience: '3.5',
    education_level: 'Bachelor',
    job_title: 'Software Engineer',
    skills: ['Git', 'SQL', 'Docker'],
    programming_languages: ['Python', 'JavaScript'],
    country: 'United States'
  });
  const [errors, setErrors] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [similarProfiles, setSimilarProfiles] = useState(null);
  const [scenarios, setScenarios] = useState(null);
  const [salaryDrivers, setSalaryDrivers] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load initial salary drivers on mount
  useEffect(() => {
    api.getSalaryDrivers()
      .then(res => setSalaryDrivers(res))
      .catch(() => {});
  }, []);

  const validateField = (name, value) => {
    switch (name) {
      case 'years_experience':
        const years = parseFloat(value);
        if (value === '') return 'Years of experience is required';
        if (isNaN(years)) return 'Must be a numeric value';
        if (years < 0) return 'Cannot be negative';
        if (years > 50) return 'Max 50 years supported';
        return '';
      case 'education_level':
        return value ? '' : 'Education level is required';
      case 'job_title':
        return value ? '' : 'Job title is required';
      case 'country':
        return value ? '' : 'Country is required';
      default:
        return '';
    }
  };

  const handleChange = (name, value) => {
    const updated = { ...formData, [name]: value };
    setFormData(updated);
    if (onProfileUpdate) onProfileUpdate(updated);
    const err = validateField(name, value);
    setErrors(prev => ({ ...prev, [name]: err }));
  };

  const validateForm = () => {
    const newErrors = {};
    let hasErrors = false;
    Object.keys(formData).forEach(key => {
      const err = validateField(key, formData[key]);
      if (err) {
        newErrors[key] = err;
        hasErrors = true;
      }
    });
    setErrors(newErrors);
    return !hasErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setLoading(true);
    setError(null);

    const input = {
      years_experience: parseFloat(formData.years_experience),
      education_level: formData.education_level,
      job_title: formData.job_title,
      skills: formData.skills,
      programming_languages: formData.programming_languages,
      country: formData.country
    };

    try {
      // Parallel fetch for prediction, similar profiles, and scenarios
      const [predRes, similarRes, scenarioRes] = await Promise.all([
        api.predictSalary(input),
        api.getSimilarProfiles(input).catch(err => ({ profiles: [], disclosure: err.message })),
        api.getCareerScenarios(input).catch(() => ({ scenarios: [] }))
      ]);

      setPrediction(predRes);
      setSimilarProfiles(similarRes);
      setScenarios(scenarioRes.scenarios || []);
      if (onProfileUpdate) onProfileUpdate(input);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
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

  return (
    <div className="predictor-container">
      {/* Input Form Card */}
      <div className="card">
        <div className="card-header-flex">
          <div>
            <h2>Professional Profile & Salary Predictor</h2>
            <p className="text-muted">
              Enter your qualifications to receive an official ML salary estimate, interpretable linear baseline, scenario projections, and similar benchmark profiles.
            </p>
          </div>
          {modelComparison?.selected_model && (
            <span className="badge badge-selected" title={modelComparison.selection_criteria}>
              Active Estimator: {modelComparison.selected_model.replace('_', ' ').toUpperCase()}
            </span>
          )}
        </div>

        {error && <div className="alert alert-error" role="alert">{error}</div>}

        <form onSubmit={handleSubmit} className="predictor-form">
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="years_experience">Years of Experience *</label>
              <input
                type="number"
                id="years_experience"
                step="0.5"
                min="0"
                max="50"
                value={formData.years_experience}
                onChange={(e) => handleChange('years_experience', e.target.value)}
                className={errors.years_experience ? 'error' : ''}
                placeholder="e.g. 4.0"
              />
              {errors.years_experience && <span className="error-message">{errors.years_experience}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="education_level">Education Level *</label>
              <select
                id="education_level"
                value={formData.education_level}
                onChange={(e) => handleChange('education_level', e.target.value)}
                className={errors.education_level ? 'error' : ''}
              >
                <option value="">Select Education Level</option>
                {EDUCATION_LEVELS.map(lvl => (
                  <option key={lvl} value={lvl}>{lvl}</option>
                ))}
              </select>
              {errors.education_level && <span className="error-message">{errors.education_level}</span>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="job_title">Job Title / Role *</label>
              <select
                id="job_title"
                value={formData.job_title}
                onChange={(e) => handleChange('job_title', e.target.value)}
                className={errors.job_title ? 'error' : ''}
              >
                <option value="">Select Job Title</option>
                {JOB_TITLES.map(title => (
                  <option key={title} value={title}>{title}</option>
                ))}
              </select>
              {errors.job_title && <span className="error-message">{errors.job_title}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="country">Country *</label>
              <select
                id="country"
                value={formData.country}
                onChange={(e) => handleChange('country', e.target.value)}
                className={errors.country ? 'error' : ''}
              >
                <option value="">Select Country</option>
                {COUNTRIES.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
              {errors.country && <span className="error-message">{errors.country}</span>}
            </div>
          </div>

          <div className="form-group">
            <label>Technical Skills</label>
            <MultiSelect
              options={ALL_SKILLS}
              selected={formData.skills}
              onChange={(s) => handleChange('skills', s)}
              placeholder="Select technical skills (Docker, AWS, Kubernetes, etc.)"
            />
            <span className="helper-text">Multi-hot encoded into candidate feature vectors.</span>
          </div>

          <div className="form-group">
            <label>Programming Languages</label>
            <MultiSelect
              options={ALL_LANGUAGES}
              selected={formData.programming_languages}
              onChange={(l) => handleChange('programming_languages', l)}
              placeholder="Select languages (Python, Go, JavaScript, etc.)"
            />
            <span className="helper-text">High-concurrency and backend languages mapped into feature matrix.</span>
          </div>

          <div className="form-actions">
            <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
              {loading ? 'Evaluating Models...' : 'Calculate Salary Estimate'}
            </button>
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={() => {
                setFormData({
                  years_experience: '',
                  education_level: '',
                  job_title: '',
                  skills: [],
                  programming_languages: [],
                  country: ''
                });
                setErrors({});
                setPrediction(null);
                setSimilarProfiles(null);
                setScenarios(null);
              }}
              disabled={loading}
            >
              Reset
            </button>
          </div>
        </form>
      </div>

      {/* Prediction & Baseline Results */}
      {prediction && (
        <div className="prediction-hero-grid">
          {/* Primary Model Card */}
          <div className="card hero-card primary-estimate-card">
            <div className="card-tag">OFFICIAL PRODUCTION ESTIMATE</div>
            <div className="hero-amount">{formatCurrency(prediction.predicted_salary)}</div>
            <div className="hero-subtext">{prediction.currency} / Year Base Compensation</div>
            
            <div className="hero-meta-box">
              <div>
                <span className="meta-label">Selected Model:</span>
                <strong> {prediction.model_name.replace('_', ' ').toUpperCase()}</strong>
              </div>
              <div>
                <span className="meta-label">Artifact Version:</span>
                <code> {prediction.model_version}</code>
              </div>
              <div className="meta-note">
                Strictly selected based on lowest Validation MAE across 5 identically preprocessed candidate algorithms.
              </div>
            </div>

            <p className="disclaimer-text">{prediction.disclaimer}</p>

            <button 
              className="btn btn-success btn-block"
              style={{ marginTop: '16px' }}
              onClick={() => onNavigateToCareerInsights && onNavigateToCareerInsights(formData, prediction.predicted_salary)}
            >
              Get AI Career Recommendations for this Estimate →
            </button>
          </div>

          {/* Interpretable Linear Baseline Card */}
          <div className="card hero-card baseline-card">
            <div className="card-tag baseline-tag">INTERPRETABLE LINEAR BASELINE</div>
            <div className="hero-amount baseline-amount">
              {prediction.baseline_salary ? formatCurrency(prediction.baseline_salary) : 'Computing...'}
            </div>
            <div className="hero-subtext">Multiple Linear Regression Reference</div>

            {prediction.baseline_salary && (
              <div className="baseline-diff-box">
                <div className="diff-row">
                  <span>Delta vs. Production Model:</span>
                  <strong className={prediction.baseline_diff >= 0 ? 'text-positive' : 'text-negative'}>
                    {prediction.baseline_diff >= 0 ? '+' : ''}{formatCurrency(prediction.baseline_diff)} ({prediction.baseline_diff_pct >= 0 ? '+' : ''}{prediction.baseline_diff_pct}%)
                  </strong>
                </div>
                <p className="diff-explanation">
                  Linear regression provides an interpretable additive baseline ($y = \sum w_i x_i$). 
                  The delta highlights non-linear feature interactions and split thresholds captured by your selected ensemble estimator.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Salary Drivers Analysis */}
      {salaryDrivers && salaryDrivers.categories && (
        <div className="card">
          <h2>Feature Driver Breakdown (Random Forest Gini Importance)</h2>
          <p className="text-muted">
            Aggregated relative importance showing how different feature groups contribute to compensation predictions.
          </p>

          <div className="driver-bars-grid">
            {Object.entries(salaryDrivers.categories).map(([cat, pct]) => (
              <div key={cat} className="driver-bar-item">
                <div className="driver-bar-header">
                  <span className="driver-name">{cat}</span>
                  <span className="driver-pct">{pct}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.min(100, pct)}%` }}></div>
                </div>
              </div>
            ))}
          </div>

          {salaryDrivers.top_drivers?.length > 0 && (
            <div className="top-drivers-section">
              <h3>Top Influential Features</h3>
              <div className="top-driver-tags">
                {salaryDrivers.top_drivers.slice(0, 6).map((d, i) => (
                  <span key={i} className="driver-pill">
                    <strong>#{i + 1} {d.name}</strong>: {d.importance_pct}%
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Career Scenario Explorer */}
      {scenarios && scenarios.length > 0 && (
        <div className="card">
          <h2>Career Scenario Explorer</h2>
          <p className="text-muted">
            Projected salary trajectories evaluated through continuous model weights and decision tree partition rules.
          </p>

          <div className="scenarios-grid">
            {scenarios.map((sc, idx) => {
              const isPositive = sc.delta >= 0;
              return (
                <div key={idx} className="scenario-card">
                  <div className="scenario-header">
                    <h4>{sc.title}</h4>
                    <span className={`delta-badge ${isPositive ? 'delta-positive' : 'delta-neutral'}`}>
                      {isPositive ? '+' : ''}{formatCurrency(sc.delta)} ({isPositive ? '+' : ''}{sc.delta_pct}%)
                    </span>
                  </div>
                  <p className="scenario-desc">{sc.description}</p>
                  <div className="scenario-salary">
                    Projected: <strong>{formatCurrency(sc.new_salary)}</strong>
                  </div>
                  <div className="scenario-note">
                    <small>Rule: {sc.rule_note}</small>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Similar Profile Insights */}
      {similarProfiles && similarProfiles.profiles && similarProfiles.profiles.length > 0 && (
        <div className="card">
          <div className="card-header-flex">
            <div>
              <h2>Similar Profile Insights (Nearest Neighbors)</h2>
              <p className="text-muted">
                Profiles from the active benchmark dataset closest in feature space (target salary excluded from distance calculation).
              </p>
            </div>
            <span className="badge badge-info">
              {similarProfiles.benchmark_count} Benchmark Records
            </span>
          </div>

          <div className="similar-profiles-grid">
            {similarProfiles.profiles.map((item, i) => (
              <div key={i} className="similar-profile-card">
                <div className="similar-card-header">
                  <span className="profile-role">
                    {item.profile.job_title}
                  </span>
                  <span className="similarity-badge">
                    {item.similarity_score}% Match
                  </span>
                </div>

                <div className="profile-stats">
                  <div><strong>Experience:</strong> {item.profile.years_experience} yrs</div>
                  <div><strong>Education:</strong> {item.profile.education_level}</div>
                  <div><strong>Country:</strong> {item.profile.country}</div>
                  <div><strong>Benchmark Salary:</strong> {formatCurrency(item.benchmark_salary)}</div>
                </div>

                <div className="profile-skills-preview">
                  <small><strong>Skills:</strong> {
                    Array.isArray(item.profile.skills) 
                      ? item.profile.skills.slice(0, 4).join(', ')
                      : String(item.profile.skills)
                  }</small>
                </div>
              </div>
            ))}
          </div>

          <div className="disclosure-box">
            <small>{similarProfiles.disclosure}</small>
          </div>
        </div>
      )}
    </div>
  );
}

export default SalaryPredictor;
