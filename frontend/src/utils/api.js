const API_BASE = '/api';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = { ...options };
  const headers = { ...options.headers };

  if (config.body instanceof FormData) {
    // Crucial fix: Let the browser set Content-Type with the multipart boundary
    delete headers['Content-Type'];
  } else {
    if (!headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
    if (config.body && typeof config.body === 'object') {
      config.body = JSON.stringify(config.body);
    }
  }

  config.headers = headers;

  try {
    const response = await fetch(url, config);
    let data;
    try {
      data = await response.json();
    } catch {
      data = { detail: response.statusText };
    }
    
    if (!response.ok) {
      const errorMsg = data?.detail || `HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMsg);
    }
    
    return data;
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error('Cannot connect to backend. Is the server running on port 8000?');
    }
    throw err;
  }
}

export const api = {
  getHealth: () => request('/health'),
  
  getDatasetInfo: () => request('/dataset/info'),
  
  getModelComparison: () => request('/models/comparison'),
  
  predictSalary: (input) => request('/predict', {
    method: 'POST',
    body: input
  }),
  
  getSimilarProfiles: (input) => request('/insights/similar-profiles', {
    method: 'POST',
    body: input
  }),
  
  getCareerScenarios: (input) => request('/insights/scenarios', {
    method: 'POST',
    body: input
  }),
  
  getSalaryDrivers: () => request('/insights/salary-drivers'),
  
  validateDataset: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/dataset/upload/validate', {
      method: 'POST',
      body: formData
    });
  },
  
  retrainModel: (file, confirm = false) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confirm', confirm.toString());
    return request('/models/retrain', {
      method: 'POST',
      body: formData
    });
  },
  
  getTrainingHistory: () => request('/models/history'),
  
  activateModel: (version) => request('/models/activate', {
    method: 'POST',
    body: { version }
  }),
  
  getCareerRecommendations: (data) => request('/recommendations', {
    method: 'POST',
    body: data
  })
};