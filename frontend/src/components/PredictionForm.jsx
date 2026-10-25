import React, { useState, useEffect } from 'react';
import { api } from '../utils/api';
import MultiSelect from './MultiSelect';
import CareerRecommendations from './CareerRecommendations';

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

function PredictionForm({ datasetInfo, modelComparison }) {
  const [formData, setFormData] = useState({
    years_experience: '',
    education_level: '',
    job_title: '',
    skills: [],
    programming_languages: [],
    country: ''
  });
  const [errors, setErrors] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showRecommendations, setShowRecommendations] = useState(false);

  // Get model name for display
  const modelName = modelComparison?.selected_model 
    ? modelComparison.models.find(m => m.algorithm === modelComparison.selected_model)?.model_name
    : 'Not available';

  const validateField = (name, value) => {
    switch (name) {
      case 'years_experience':
        const years = parseFloat(value);
        if (value === '') return 'Years of experience is required';
        if (isNaN(years)) return 'Must be a number';
        if (years < 0) return 'Cannot be negative';
        if (years > 50) return 'Seems unrealistic (>50 years)';
        return '';
      case 'education_level':
        if (!value) return 'Education level is required';
        return '';
      case 'job_title':
        if (!value) return 'Job title is required';
        return '';
      case 'country':
        if (!value) return 'Country is required';
        return '';
      default:
        return '';
    }
  };

  const handleChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
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
    if (!healthCheck()) return;

    setLoading(true);
    setError(null);

    try {
      const input = {
        years_experience: parseFloat(formData.years_experience),
        education_level: formData.education_level,
        job_title: formData.job_title,
        skills: formData.skills,
        programming_languages: formData.programming_languages,
        country: formData.country
      };
      
      const result = await api.predictSalary(input);
      setPrediction(result);
      setShowRecommendations(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const healthCheck = () => {
    if (!modelComparison) {
      setError('No trained model available. Please train a model first.');
      return false;
    }
    return true;
  };

  const handleRecommendations = async () => {
    if (!prediction) return;
    setShowRecommendations(true);
  };

  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }).format(val);
  };

  return (
    <div>
      <div className="card">
        <h2>Salary Prediction</h2>
        <p style={{ marginBottom: '20px', color: '#7f8c8d' }}>
          Enter your professional profile to get an estimated annual salary.
          {modelComparison && (
            <span className="badge badge-selected" style={{ marginLeft: '8px' }}>
              Active Model: {modelName}
            </span>
          )}
        </p>

        {error && <div className="alert alert-error" role="alert">{error}</div>}

        <form onSubmit={handleSubmit}>
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
                placeholder="e.g., 3.5"
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
                <option value="">Select...</option>
                {EDUCATION_LEVELS.map(level => (
                  <option key={level} value={level}>{level}</option>
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
                <option value="">Select...</option>
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
                <option value="">Select...</option>
                {COUNTRIES.map(country => (
                  <option key={country} value={country}>{country}</option>
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
              onChange={(skills) => handleChange('skills', skills)}
              placeholder="Select skills (Git, Docker, AWS, Kubernetes, etc.)"
              maxVisible={5}
            />
            <span className="helper-text">Select all that apply. These are multi-hot encoded for the model.</span>
          </div>

          <div className="form-group">
            <label>Programming Languages</label>
            <MultiSelect
              options={ALL_LANGUAGES}
              selected={formData.programming_languages}
              onChange={(languages) => handleChange('programming_languages', languages)}
              placeholder="Select languages (Python, JavaScript, Java, Go, etc.)"
              maxVisible={5}
            />
            <span className="helper-text">Select all that apply. These are multi-hot encoded for the model.</span>
          </div>

          <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Predict Salary'}
            </button>
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={() => { setFormData({ years_experience: '', education_level: '', job_title: '', skills: [], programming_languages: [], country: '' }); setErrors({}); setPrediction(null); }}
              disabled={loading}
            >
              Clear Form
            </button>
          </div>
        </form>

        {prediction && (
          <div className="prediction-result">
            <div className="prediction-amount">{formatCurrency(prediction.predicted_salary)}</div>
            <div className="prediction-currency">{prediction.currency} (Annual)</div>
            <div className="prediction-meta">
              <span>Model: <strong>{prediction.model_name}</strong></span>
              <span>Version: <strong>{prediction.model_version}</strong></span>
            </div>
            <p className="prediction-disclaimer">{prediction.disclaimer}</p>
            
            <div style={{ marginTop: '20px', display: 'flex', gap: '12px', justifyContent: 'center' }}>
              <button 
                className="btn btn-success"
                onClick={handleRecommendations}
                disabled={loading}
              >
                Get AI Career Recommendations
              </button>
            </div>
          </div>
        )}

        {showRecommendations && (
          <CareerRecommendations
            profile={formData}
            predictedSalary={prediction?.predicted_salary}
            currency={prediction?.currency}
            modelName={prediction?.model_name}
            onClose={() => setShowRecommendations(false)}
          />
        )}
      </div>

      {/* Dataset Info Panel */}
      {datasetInfo && (
        <div className="card">
          <h2>Dataset Information</h2>
          <div className="grid-2">
            <div>
              <h3>Current Training Data</h3>
              <p><strong>Records:</strong> {datasetInfo.row_count}</p>
              <p><strong>Features:</strong> {datasetInfo.column_count - 1} inputs + 1 target</p>
              <p><strong>Type:</strong> {datasetInfo.is_synthetic ? 'Synthetic (Educational)' : 'Real-world'}</p>
              <p><strong>Source:</strong> {datasetInfo.source}</p>
            </div>
            <div>
              <h3>Supported Categories</h3>
              <p><strong>Education:</strong> {EDUCATION_LEVELS.join(', ')}</p>
              <p><strong>Countries:</strong> {COUNTRIES.join(', ')}</p>
              <p><strong>Job Titles:</strong> {JOB_TITLES.length} roles available</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;