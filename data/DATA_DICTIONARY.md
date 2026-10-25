# Data Dictionary — Salary Prediction Dataset

## Overview
This is a **synthetic educational dataset** created for demonstration and learning purposes. It does not represent real employee records or actual salary surveys. All values are artificially generated to illustrate typical patterns in software engineering compensation.

## Columns

| Column | Type | Description | Unit | Role |
|--------|------|-------------|------|------|
| `years_experience` | float | Years of professional work experience | years | Feature (numeric) |
| `education_level` | string | Highest completed education level | category | Feature (categorical) |
| `job_title` | string | Job role / title | category | Feature (categorical) |
| `skills` | string (JSON array) | Technical skills and tools | list of strings | Feature (categorical, multi-label) |
| `programming_languages` | string (JSON array) | Programming languages known | list of strings | Feature (categorical, multi-label) |
| `country` | string | Country of employment | category | Feature (categorical) |
| `annual_salary_usd` | float | Annual salary normalized to USD | USD | Target (regression) |

## Education Levels (ordered)
1. High School
2. Bachelor
3. Master
4. PhD

## Job Titles
- Junior Developer
- Software Engineer
- Senior Software Engineer
- Lead Engineer
- Principal Engineer
- Data Analyst
- Data Scientist
- Senior Data Scientist
- Principal Data Scientist
- DevOps Engineer
- Senior DevOps Engineer
- Principal DevOps Engineer

## Countries
- United States
- India
- Germany
- United Kingdom
- Canada
- Australia

## Skills (examples)
Git, SQL, Docker, AWS, Kubernetes, CI/CD, Terraform, Microservices, Architecture, Mentoring, Research, Strategy, Excel, Tableau, PowerBI, ML, Deep Learning, MLOps, Monitoring, Bash

## Programming Languages (examples)
Python, JavaScript, TypeScript, Java, Go, Rust, C++, Scala, Haskell, R, SQL, Bash

## Limitations
- **Synthetic data**: Patterns are simplified and do not capture real-world variance, negotiation effects, company-specific policies, or market fluctuations.
- **Currency normalization**: All salaries converted to USD at fixed rates; real data would have currency volatility.
- **Missing features**: No company size, industry, remote/hybrid/onsite, equity/bonus, cost-of-living adjustment, gender, age, or negotiation history.
- **Small sample**: ~100 records; not statistically representative.
- **No temporal component**: Data does not reflect salary trends over time.
- **Multi-label encoding**: Skills and programming languages stored as JSON arrays; require special preprocessing (multi-hot encoding).

## Usage Notes
- This dataset is provided for **educational demonstration only**.
- Do not use for real salary negotiations, benchmarking, or employment decisions.
- The model trained on this data demonstrates ML pipeline functionality, not production-grade salary prediction.