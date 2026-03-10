# 🗂️ Project Plan — Industry-Simulated AI Product Workflow
> Solo student project · AI agents = CrewAI agents
> Stack: CrewAI · Python · Flask/Streamlit · Pandas · Scikit-Learn · Matplotlib/Seaborn · GitHub

---

## Phase 0 — Project Setup

- [x] 0.1 Create a GitHub repository (e.g. `ai-product-workflow`)
- [x] 0.2 Create the folder structure:
  ```
  /
  ├── data/              # raw & cleaned datasets
  ├── outputs/           # all required output artifacts
  ├── crew_analyst/      # Data Analyst Crew
  ├── crew_scientist/    # Data Scientist Crew
  ├── flow/              # CrewAI Flow orchestration
  ├── app/               # Streamlit / Flask UI
  ├── logs/              # run logs
  └── docs/              # documentation
  ```
- [x] 0.3 Create `requirements.txt` (`crewai`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`/`flask`, `ydata-profiling`, etc.)
- [x] 0.4 Create `.gitignore` (`__pycache__`, `.env`, `*.env`)
- [x] 0.5 Create a root `README.md` (project description + how to run)
- [x] 0.6 Set up Python virtual environment and install dependencies

---

## Phase 1 — Dataset Selection & Ingestion

- [x] 1.1 Choose a dataset from Kaggle or public open data (retail/sales/customer domain) — `mashlyn/online-retail-ii-uci` (Online Retail II, ~1M rows)
- [x] 1.2 Place raw dataset in `/data/raw/` — `online_retail_II.csv` (90.5 MB)
- [x] 1.3 Commit download script `scripts/download_dataset.py` (dataset too large for git)
- [x] 1.4 Write a brief dataset description in `docs/dataset_description.md`

---

## Phase 2 — Crew 1: Data Analyst Crew (≥ 3 AI Agents)

### 2.1 Define the AI Agents (`crew_analyst/agents.py`)

- [x] 2.1.1 **Agent 1 — Data Ingestor & Validator**: loads raw data, checks schema, detects nulls / duplicates / type errors
- [x] 2.1.2 **Agent 2 — Data Cleaner**: handles nulls, fixes dtypes, removes duplicates, logs every transformation
- [x] 2.1.3 **Agent 3 — EDA & Insights Analyst**: descriptive stats, correlations, visualizations, business summaries
- [x] 2.1.4 **Agent 4 — Dataset Contract Author** *(optional 4th agent)*: formalizes schema, allowed ranges, constraints into JSON

### 2.2 Define Tasks (`crew_analyst/tasks.py`)

- [x] 2.2.1 Task for Agent 1: validate raw data, report issues
- [x] 2.2.2 Task for Agent 2: clean data, save `clean_data.csv`
- [x] 2.2.3 Task for Agent 3: run EDA, generate `eda_report.html` and `insights.md`
- [x] 2.2.4 Task for Agent 4: produce `dataset_contract.json`

### 2.3 Assemble & Run Crew (`crew_analyst/crew.py`)

- [x] 2.3.1 Instantiate `Crew` with agents and tasks
- [x] 2.3.2 Run the crew and verify all 4 outputs are produced

### 2.4 Required Outputs

- [x] `/outputs/clean_data.csv`
- [x] `/outputs/eda_report.html`
- [x] `/outputs/insights.md` (≥ 5 business findings)
- [x] `/outputs/dataset_contract.json` (column names, dtypes, allowed values, nullability, constraints)

---

## Phase 3 — Crew 2: Data Scientist Crew (≥ 3 AI Agents)

### 3.1 Define the AI Agents (`crew_scientist/agents.py`)

- [x] 3.1.1 **Agent 1 — Contract Validator**: loads `clean_data.csv`, validates against `dataset_contract.json`, fails gracefully on violations
- [x] 3.1.2 **Agent 2 — Feature Engineer**: encodes categoricals, scales numerics, creates new features, saves `features.csv`
- [x] 3.1.3 **Agent 3 — Model Trainer & Evaluator**: trains ≥ 2 model variations, compares metrics, saves best as `model.pkl`
- [x] 3.1.4 **Agent 4 — Model Card Author** *(optional 4th agent)*: writes purpose, training summary, metrics, limitations, ethics

### 3.2 Define Tasks (`crew_scientist/tasks.py`)

- [x] 3.2.1 Task for Agent 1: validate contract vs. cleaned data
- [x] 3.2.2 Task for Agent 2: feature engineering, save `features.csv`
- [x] 3.2.3 Task for Agent 3: train both models, compare, save `model.pkl` + `evaluation_report.md`
- [x] 3.2.4 Task for Agent 4: write `model_card.md`

### 3.3 Assemble & Run Crew (`crew_scientist/crew.py`)

- [x] 3.3.1 Instantiate `Crew` with agents and tasks
- [x] 3.3.2 Run the crew and verify all 4 outputs are produced

### 3.4 Required Outputs

- [x] `/outputs/features.csv`
- [x] `/outputs/model.pkl` (or `.joblib`)
- [x] `/outputs/evaluation_report.md` (metrics table for both models + winner rationale)
- [x] `/outputs/model_card.md` (purpose · training summary · metrics · limitations · ethics)

---

## Phase 4 — CrewAI Flow Orchestration (`flow/main_flow.py`)

- [x] 4.1 Define Flow using `@start` and `@listen` decorators
- [x] 4.2 **Step 1**: trigger Analyst Crew, await all 4 outputs
- [x] 4.3 **Validation Gate 1**: verify `dataset_contract.json` matches schema of `clean_data.csv`
  - Fail gracefully (log error + raise informative exception) if mismatch
- [x] 4.4 **Step 2**: trigger Scientist Crew, passing Analyst outputs
- [x] 4.5 **Validation Gate 2**: confirm all required features exist in `features.csv` before modeling
  - Fail gracefully if features are missing
- [x] 4.6 Add structured logging (timestamp · step name · success/failure) to `/logs/`
- [x] 4.7 Ensure deterministic steps: fixed `random_state=42`, versioned data paths
- [x] 4.8 Run the full Flow end-to-end and confirm both crews complete successfully
- [x] 4.9 Test graceful failure: corrupt the contract → confirm Gate 1 catches it
- [x] 4.10 Test graceful failure: remove a feature → confirm Gate 2 catches it

---

## Phase 5 — Web Application (`app/`)

- [x] 5.1 Choose interface: Streamlit (simpler) or Flask (more control) — **Streamlit**
- [x] 5.2 Build pages/sections:
  - **Run Flow** — trigger the full pipeline with a button
  - **EDA Report** — display `eda_report.html` inline
  - **Predict** — input feature values → model returns a prediction
  - **Downloads** — links to all output artifacts
- [x] 5.3 Load `model.pkl` for inference in the UI
- [x] 5.4 Test the app locally end-to-end
- [ ] 5.5 (Optional) Deploy to Streamlit Cloud or Railway, add live URL to `README.md`

---

## Phase 6 — Reproducibility & Documentation

- [x] 6.1 Confirm `random_state=42` (or equivalent) is set in all model/split calls
- [x] 6.2 Use Python `logging` module for all major steps
- [x] 6.3 Verify all artifacts are committed under `/outputs/` and `/logs/`
- [ ] 6.4 Finalize `README.md`: setup · how to run Flow · how to run app · output descriptions
- [x] 6.5 Add docstrings and inline comments throughout

---

## Phase 7 — Final Deliverables

- [x] 7.1 Push everything to GitHub and confirm the repo is clean and complete
- [x] 7.2 Create **business presentation** (10–12 slides):
  - Slide 1: Title + dataset overview
  - Slide 2: Architecture diagram (Flow → Crew 1 → validation → Crew 2)
  - Slides 3–4: Analyst Crew — EDA highlights & key findings
  - Slides 5–6: Scientist Crew — features, models, metrics comparison
  - Slide 7: Model card summary (limitations + ethics)
  - Slide 8: App demo screenshots
  - Slides 9–10: Challenges, lessons learned, future improvements
- [ ] 7.3 Record **demo video (≤ 5 min)**:
  - Walk through repo structure
  - Show Flow running (terminal output)
  - Demo the UI (make a prediction)
  - Show key artifacts (`eda_report.html`, `model_card.md`)
  - Upload to YouTube/Drive and link in `README.md`

---

## ✅ Output Artifacts Checklist

| File | Produced by | Path |
|---|---|---|
| `clean_data.csv` | Analyst Crew | `/outputs/` |
| `eda_report.html` | Analyst Crew | `/outputs/` |
| `insights.md` | Analyst Crew | `/outputs/` |
| `dataset_contract.json` | Analyst Crew | `/outputs/` |
| `features.csv` | Scientist Crew | `/outputs/` |
| `model.pkl` | Scientist Crew | `/outputs/` |
| `evaluation_report.md` | Scientist Crew | `/outputs/` |
| `model_card.md` | Scientist Crew | `/outputs/` |

---

## 🔑 Key Constraints

- Crew 1 & Crew 2 must each have **≥ 3 CrewAI agents**
- The Flow must **automate the handoff** between the two crews
- Both **validation gates must fail gracefully** (no unhandled crashes)
- All steps must be **deterministic** (fixed seeds, versioned paths)
- All artifacts must be **saved inside the repo**
