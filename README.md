# AI Product Workflow

> Solo student project — Industry-simulated AI product pipeline using **CrewAI agents**

## 🗂️ Project Overview

This project simulates a real-world AI product development workflow using two coordinated CrewAI crews:

1. **Crew 1 — Data Analyst Crew**: Ingests, validates, cleans, and analyzes raw data, producing EDA reports and a dataset contract.
2. **Crew 2 — Data Scientist Crew**: Takes cleaned data, engineers features, trains ML models, and produces evaluation reports and a model card.
3. **CrewAI Flow**: Orchestrates both crews with validation gates between steps to ensure data quality and reproducibility.

## 🛠️ Tech Stack

- **AI Agents**: [CrewAI](https://github.com/joaomdmoura/crewAI)
- **Language**: Python 3.10+
- **Data**: Pandas, NumPy, ydata-profiling
- **ML**: Scikit-Learn, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit / Flask
- **Version Control**: GitHub

## 📁 Folder Structure

```
/
├── data/
│   └── raw/               # Raw dataset(s)
├── outputs/               # All required output artifacts
├── crew_analyst/          # Data Analyst Crew (agents, tasks, crew)
├── crew_scientist/        # Data Scientist Crew (agents, tasks, crew)
├── flow/                  # CrewAI Flow orchestration
├── app/                   # Streamlit / Flask web UI
├── logs/                  # Structured run logs
└── docs/                  # Documentation & dataset description
```

## 🚀 How to Run

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/ilanafadida-hub/final-project.git
cd final-project

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (CrewAI Flow)

```bash
python flow/main_flow.py
```

### 3. Run the Web Application

```bash
streamlit run app/app.py
```

### 4. Demo Video

[▶️ Watch the 5-minute Demo Video on YouTube](https://youtube.com/placeholder)

## 📦 Output Artifacts

| File | Description | Path |
|---|---|---|
| `clean_data.csv` | Cleaned dataset | `/outputs/` |
| `eda_report.html` | EDA profiling report | `/outputs/` |
| `insights.md` | ≥ 5 business findings | `/outputs/` |
| `dataset_contract.json` | Schema + constraints | `/outputs/` |
| `features.csv` | Engineered features | `/outputs/` |
| `model.pkl` | Best trained model | `/outputs/` |
| `evaluation_report.md` | Model comparison & metrics | `/outputs/` |
| `model_card.md` | Model purpose, limits, ethics | `/outputs/` |

## 🔑 Key Design Constraints

- Each crew has **≥ 3 CrewAI agents**
- The Flow **automates the handoff** between both crews
- Both **validation gates fail gracefully** (no unhandled crashes)
- All steps are **deterministic** (`random_state=42`, versioned paths)
- All artifacts are **saved inside the repo**

## 📄 License

MIT
