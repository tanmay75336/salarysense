# SalarySense AI — Intelligent Salary Prediction & Career Recommendation System

An end-to-end, full-stack Machine Learning application and AI Career Advisory system designed for precise software engineering compensation modeling, career progression simulation, feature driver analysis, and automated dataset retraining.

---

## 🌟 Core Features & Functional Roles

SalarySense AI trains **all 5 candidate regression algorithms** on identical preprocessed feature matrices and identical data splits. A single official estimator is selected **strictly by lowest Validation MAE**, while the remaining candidate models fulfill distinct, non-redundant functional roles across the application:

1. **Feature 1: Primary Salary Estimate (Selected Production Estimator)**
   - Selected strictly by lowest Validation MAE (with held-out test data strictly isolated until final verification).
   - Inferences include active algorithm badge, artifact version hash, and benchmark confidence context.

2. **Feature 2: Similar Profile Insights (Nearest Neighbors Index)**
   - Fits a `NearestNeighbors` model on normalized feature representations of the training benchmark.
   - **Target salary is strictly excluded** from distance computations to prevent target leakage.
   - Computes cosine similarity match percentages and provides explicit benchmark disclosure notes.

3. **Feature 3: Career Scenario Explorer (Ensemble & Decision Tree Split Rules)**
   - Evaluates what-if career progression scenarios:
     * Experience growth (+2 years, +4 years)
     * Degree advancement (Bachelor → Master → PhD)
     * High-demand cloud & infrastructure upskilling (AWS, Docker, Kubernetes)
     * High-concurrency languages (Go, TypeScript)
   - Projects new salary, dollar delta, and percentage gains backed by tree partition logic.

4. **Feature 4: Interpretable Salary Baseline (Multiple Linear Regression)**
   - Computes an additive linear baseline estimate ($y = \sum w_i x_i$).
   - Calculates real-time delta ($ and %) against the primary model, highlighting non-linear feature interactions and split thresholds captured by the selected ensemble.

5. **Feature 5: Salary Driver Analysis (Random Forest Feature Importance)**
   - Aggregates Random Forest Mean Decrease in Impurity (MDI) across high-level categories:
     * Years of Experience
     * Job Title
     * Education Level
     * Country & Region
     * Technical Skills
     * Programming Languages
   - Visualizes live progress tracks and top influential feature drivers.

6. **Feature 6: Decoupled AI Career Advisor (Groq LLaMA-3 Integration)**
   - Decoupled from numeric predictions (never alters the ML estimate).
   - Uses `GROQ_API_KEY` from `backend/.env` with automatic candidate model fallbacks (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).
   - Produces structured JSON: prioritized learning roadmaps (High/Med/Low), in-demand skills, demonstrable portfolio projects, recognized certifications, and methodological boundaries.

---

## 🛠️ CSV Ingestion, Validation & Scalability

- **Complete Multipart Upload Repair**: Fixed client `FormData` header construction so browsers automatically generate correct multipart boundaries without triggering 422 errors.
- **Robust Column Alias Normalization**: Automatically standardizes common header variants (e.g., `experience_years` → `years_experience`, `salary` → `annual_salary_usd`, `role` → `job_title`).
- **Detailed Ingestion Diagnostics**: Every upload reports:
  * Total Input Rows
  * Accepted Clean Rows
  * Duplicate Rows (vectorized deduplication)
  * Rejected / Invalid Rows
  * Breakdown of exclusion reasons (negative experience, non-positive salary, unparseable multi-labels)
- **Scalability**: Scalable from 10 rows to 50,000+ rows with $O(1)$ dictionary multi-label encoding and adaptive CV fold scaling.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               React + Vite Frontend (Port 3000)         │
│  - Tab 1: Salary Predictor (Estimate, Baseline, Scenarios)│
│  - Tab 2: Career Insights (Groq AI Personalized Advisor)│
│  - Tab 3: Dataset & Models (Upload, Comparison, History)│
└────────────────────────────┬────────────────────────────┘
                             │ REST API (/api/*)
┌────────────────────────────▼────────────────────────────┐
│                  FastAPI Backend (Port 8000)            │
│  - ML Pipeline: DataPreprocessor, ModelTrainer          │
│  - ModelRegistry: Versioning & Atomic Model Activation │
│  - DatasetService: Alias Mapping & Row Diagnostics      │
│  - GroqService: Decoupled AI Career Advisor            │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Running Locally

### Backend Setup
```powershell
cd backend
python -m pip install -r requirements.txt
# Ensure backend/.env contains your GROQ_API_KEY
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Frontend Setup
```powershell
cd frontend
npm install
npm run build
npm run dev -- --port 3000
```

- **Frontend Application:** http://localhost:3000
- **FastAPI Interactive Docs:** http://localhost:8000/docs
- **Health Check Endpoint:** http://localhost:8000/api/health

---

## 🧪 Comprehensive Automated Testing

### Backend Test Suite (Pytest)
```powershell
cd backend
python -m pytest tests/test_all.py tests/test_features.py -v
```
**Results:** **41 / 41 Tests Passed**
- Preprocessing & $O(1)$ MultiLabelEncoder: 5 tests
- Dataset Validation & Vectorized Diagnostics: 7 tests
- Candidate Models (all 5) & Validation MAE Selection: 6 tests
- Nearest Neighbors & Target Salary Leakage Isolation: 2 tests
- Scenario Explorer & Driver Aggregation: 3 tests
- Model Registry, Save, Rollback & Activation: 6 tests
- Groq AI Service & JSON parsing: 3 tests
- API Endpoints & CSV upload workflow: 9 tests

### Frontend Test Suite (Vitest)
```powershell
cd frontend
npm test -- --run
```
**Results:** **6 / 6 Tests Passed**
- Multipart/form-data upload boundary verification (confirms `Content-Type` is never overridden with `application/json`)
- Retrain model FormData confirm payload
- JSON payload prediction headers
- Similar profiles, career scenarios, and salary driver endpoints

---

## 📊 3-Tab Interface Structure

1. **Tab 1: Salary Predictor**
   - Professional profile form (YOE, Education, Job Title, Country, Skills, Languages)
   - Official Production Estimate card
   - Interpretable Linear Baseline card with delta and variance explanation
   - Salary Drivers breakdown (Random Forest Gini Importance bars + top drivers)
   - Career Scenario Explorer (+2 YOE, +4 YOE, Degree Advancement, Upskilling)
   - Similar Profile Insights (Nearest Neighbors with cosine match %)
2. **Tab 2: Career Insights**
   - Groq AI Career Advisor
   - Prioritized roadmap (High, Medium, Low tags)
   - In-demand skills tags
   - Proof-of-work portfolio project ideas
   - Industry recognized certifications & strategic suggestions
   - Limitations disclosure
3. **Tab 3: Dataset & Models**
   - Drag & drop CSV upload dropzone
   - Exact row summary diagnostics (Total, Accepted, Duplicate, Rejected, Reasons)
   - 5-Candidate Model Comparison Table with Validation MAE badges
   - Training history table with one-click version activation
   - Active benchmark dataset schema and limitations

---

## ⚖️ Limitations & Disclaimers

The baseline models are trained on educational benchmark data for demonstration purposes. Estimates reflect central tendencies for base compensation and do not account for discretionary bonuses, equity grants, healthcare packages, or individual negotiation.