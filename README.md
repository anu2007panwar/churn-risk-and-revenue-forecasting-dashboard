# ChurnGuard — Client Churn Risk & Revenue Forecasting

## Problem
Predict churn for 200 enterprise IT clients (from 10M transaction records),
find top churn drivers, forecast 12-month revenue, and suggest strategies to
reduce churn by 5%.

---

## Architecture

```
┌─────────────────────────────────┐      ┌───────────────────────┐
│   Frontend (index.html)         │ ────▶│  FastAPI Backend      │
│   - Overview Dashboard          │      │  localhost:8000        │
│   - Client Risk Table           │      │  /api/summary          │
│   - 12-Month Revenue Forecast   │      │  /api/clients          │
│   - Churn Factor Analysis       │      │  /api/feature-importance│
│   - Model Performance           │      │  /api/revenue-forecast  │
└─────────────────────────────────┘      │  /api/churn-factors    │
                                         │  /api/model-metrics    │
                                         │  /api/industry-breakdown│
                                         └───────────────────────┘
```

---

## Backend Setup

```bash
cd backend/

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API docs available at:
# http://localhost:8000/docs
```

---

## Frontend Setup

```bash
cd frontend/

# Option 1: Open directly in browser
open index.html

# Option 2: Serve with Python
python -m http.server 3000
# Then visit http://localhost:3000
```

---

## ML Pipeline

### Data Constraints Handled

| Constraint               | Solution                                      |
|--------------------------|-----------------------------------------------|
| 10M transaction records  | Batch-aggregated to 200 client-level features |
| 12% missing financial    | Median imputation (SimpleImputer)             |
| 18% churn (imbalanced)   | SMOTE oversampling (minority → 33%)           |

### Models

1. **Random Forest Classifier** (churn prediction)
   - 200 estimators, max_depth=8, class_weight=balanced
   - Features: NPS, tenure, payment delays, escalations, engagement, etc.

2. **Gradient Boosting Regressor** (contract value imputation)
   - Fills missing financial data for 12% of records

3. **Monthly Decay Revenue Model** (12-month forecast)
   - Compares base churn scenario vs. 5% reduction scenario

### Key Features for Churn Prediction
- NPS score
- Last payment delay (days)
- Contract renewal window
- Incident escalations
- Engagement score
- Competitor contact flag
- Tenure months
- Number of services

---

## API Endpoints

| Endpoint                   | Description                              |
|----------------------------|------------------------------------------|
| GET /api/summary           | KPI metrics (churn rate, counts, NPS)    |
| GET /api/clients           | Client list with risk scores (paginated) |
| GET /api/feature-importance| Top ML features driving churn            |
| GET /api/revenue-forecast  | 12-month base vs optimistic forecast     |
| GET /api/churn-factors     | Top drivers + retention strategies       |
| GET /api/model-metrics     | AUC, F1, precision, recall               |
| GET /api/industry-breakdown| Churn risk breakdown by industry         |

---

## Churn Reduction Strategies (5% Target)

1. AI-driven early warning system with CSM alerts
2. Proactive SLA improvement for top 20 at-risk clients
3. Multi-year contract incentives (10-15% discount for 3-year)
4. Customer health scoring dashboard for weekly reviews
5. Service bundling to increase stickiness
6. Executive Sponsor program for >$100K ARR accounts
