import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from './api';

describe('Frontend API Client & FormData Upload Handling', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('validateDataset sends FormData without application/json Content-Type header', async () => {
    let capturedUrl = '';
    let capturedConfig = {};

    global.fetch = vi.fn().mockImplementation((url, config) => {
      capturedUrl = url;
      capturedConfig = config;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ is_valid: true, total_input_rows: 50, accepted_rows: 50 })
      });
    });

    const mockFile = new Blob(['years_experience,annual_salary_usd\n3,80000'], { type: 'text/csv' });
    mockFile.name = 'test.csv';

    const result = await api.validateDataset(mockFile);

    expect(capturedUrl).toBe('/api/dataset/upload/validate');
    expect(capturedConfig.method).toBe('POST');
    expect(capturedConfig.body instanceof FormData).toBe(true);
    // Crucial check: Content-Type MUST NOT be application/json so browser sets multipart boundary
    expect(capturedConfig.headers['Content-Type']).toBeUndefined();
    expect(result.is_valid).toBe(true);
  });

  it('retrainModel sends FormData with confirm flag and without application/json Content-Type', async () => {
    let capturedConfig = {};

    global.fetch = vi.fn().mockImplementation((url, config) => {
      capturedConfig = config;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ success: true, version: 'v1_test', selected_model: 'random_forest' })
      });
    });

    const mockFile = new Blob(['test'], { type: 'text/csv' });
    const result = await api.retrainModel(mockFile, true);

    expect(capturedConfig.body instanceof FormData).toBe(true);
    expect(capturedConfig.headers['Content-Type']).toBeUndefined();
    expect(result.success).toBe(true);
    expect(result.selected_model).toBe('random_forest');
  });

  it('predictSalary sends JSON payload with application/json Content-Type', async () => {
    let capturedConfig = {};

    global.fetch = vi.fn().mockImplementation((url, config) => {
      capturedConfig = config;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          predicted_salary: 125000,
          currency: 'USD',
          model_name: 'random_forest',
          baseline_salary: 118000,
          baseline_diff: 7000
        })
      });
    });

    const payload = {
      years_experience: 4,
      education_level: 'Bachelor',
      job_title: 'Software Engineer',
      skills: ['Git', 'Docker'],
      programming_languages: ['Python'],
      country: 'United States'
    };

    const res = await api.predictSalary(payload);

    expect(capturedConfig.headers['Content-Type']).toBe('application/json');
    expect(typeof capturedConfig.body).toBe('string');
    expect(res.predicted_salary).toBe(125000);
    expect(res.baseline_salary).toBe(118000);
  });

  it('getSimilarProfiles sends POST request to /api/insights/similar-profiles', async () => {
    let capturedUrl = '';
    global.fetch = vi.fn().mockImplementation((url) => {
      capturedUrl = url;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ profiles: [], benchmark_count: 64, disclosure: 'test disclosure' })
      });
    });

    await api.getSimilarProfiles({ years_experience: 3 });
    expect(capturedUrl).toBe('/api/insights/similar-profiles');
  });

  it('getCareerScenarios sends POST request to /api/insights/scenarios', async () => {
    let capturedUrl = '';
    global.fetch = vi.fn().mockImplementation((url) => {
      capturedUrl = url;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ base_salary: 100000, currency: 'USD', scenarios: [] })
      });
    });

    await api.getCareerScenarios({ years_experience: 3 });
    expect(capturedUrl).toBe('/api/insights/scenarios');
  });

  it('getSalaryDrivers sends GET request to /api/insights/salary-drivers', async () => {
    let capturedUrl = '';
    global.fetch = vi.fn().mockImplementation((url) => {
      capturedUrl = url;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ categories: {}, top_drivers: [], model_source: 'rf' })
      });
    });

    await api.getSalaryDrivers();
    expect(capturedUrl).toBe('/api/insights/salary-drivers');
  });
});
