"""
CrewAI Flow — orchestrates the full AI product pipeline.

Steps:
  1. (@start)  run_analyst_crew  — triggers Analyst Crew, awaits 4 outputs
  2. Validation Gate 1            — contract JSON vs clean_data.csv schema
  3. (@listen) run_scientist_crew — triggers Scientist Crew, awaits 4 outputs
  4. Validation Gate 2            — required feature columns in features.csv

Logging: structured entries are written to /logs/flow_<timestamp>.log
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── make project root importable ────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from crewai.flow.flow import Flow, listen, start

# ── paths ────────────────────────────────────────────────────────────────────
OUTPUTS_DIR   = _PROJECT_ROOT / "outputs"
LOGS_DIR      = _PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DATA    = OUTPUTS_DIR / "clean_data.csv"
CONTRACT      = OUTPUTS_DIR / "dataset_contract.json"
EDA_REPORT    = OUTPUTS_DIR / "eda_report.html"
INSIGHTS      = OUTPUTS_DIR / "insights.md"
FEATURES      = OUTPUTS_DIR / "features.csv"
MODEL         = OUTPUTS_DIR / "model.pkl"
EVAL_REPORT   = OUTPUTS_DIR / "evaluation_report.md"
MODEL_CARD    = OUTPUTS_DIR / "model_card.md"

REQUIRED_ANALYST_OUTPUTS = [CLEAN_DATA, EDA_REPORT, INSIGHTS, CONTRACT]
REQUIRED_SCIENTIST_OUTPUTS = [FEATURES, MODEL, EVAL_REPORT, MODEL_CARD]
REQUIRED_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]


# ── logger factory ────────────────────────────────────────────────────────────
def _make_logger() -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"flow_{ts}.log"

    logger = logging.getLogger(f"ai_flow_{ts}")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── validation helpers ────────────────────────────────────────────────────────
def _check_analyst_outputs(log: logging.Logger) -> None:
    """Raise if any required analyst output file is missing."""
    missing = [str(p) for p in REQUIRED_ANALYST_OUTPUTS if not p.exists()]
    if missing:
        msg = f"Analyst outputs missing: {missing}"
        log.error(msg)
        raise FileNotFoundError(msg)
    log.info("All analyst outputs present: %s", [p.name for p in REQUIRED_ANALYST_OUTPUTS])


def _gate1_contract_vs_data(log: logging.Logger) -> None:
    """
    Validation Gate 1 — verify dataset_contract.json is consistent with
    clean_data.csv (all contract columns must exist in the CSV header).
    Fails gracefully with an informative exception on mismatch.
    """
    log.info("GATE 1: Validating dataset_contract.json against clean_data.csv …")

    try:
        with open(CONTRACT, encoding="utf-8") as f:
            contract = json.load(f)
    except Exception as exc:
        msg = f"Gate 1 FAILED — cannot read dataset_contract.json: {exc}"
        log.error(msg)
        raise RuntimeError(msg) from exc

    contract_columns = set(contract.get("columns", {}).keys())
    if not contract_columns:
        msg = "Gate 1 FAILED — dataset_contract.json contains no column definitions."
        log.error(msg)
        raise RuntimeError(msg)

    # Read only the CSV header (no need to load 70 MB)
    try:
        with open(CLEAN_DATA, encoding="utf-8") as f:
            header_line = f.readline()
        csv_columns = set(col.strip().strip('"') for col in header_line.split(","))
    except Exception as exc:
        msg = f"Gate 1 FAILED — cannot read clean_data.csv header: {exc}"
        log.error(msg)
        raise RuntimeError(msg) from exc

    missing_in_csv = contract_columns - csv_columns
    if missing_in_csv:
        msg = (
            f"Gate 1 FAILED — columns in contract missing from clean_data.csv: "
            f"{sorted(missing_in_csv)}"
        )
        log.error(msg)
        raise RuntimeError(msg)

    log.info("Gate 1 PASSED — all %d contract columns present in clean_data.csv.",
             len(contract_columns))


def _gate2_features_check(log: logging.Logger) -> None:
    """
    Validation Gate 2 — verify all required RFM feature columns exist in
    features.csv before the model is considered usable.
    Fails gracefully with an informative exception on mismatch.
    """
    log.info("GATE 2: Validating required features in features.csv …")

    try:
        with open(FEATURES, encoding="utf-8") as f:
            header_line = f.readline()
        csv_columns = set(col.strip().strip('"') for col in header_line.split(","))
    except Exception as exc:
        msg = f"Gate 2 FAILED — cannot read features.csv header: {exc}"
        log.error(msg)
        raise RuntimeError(msg) from exc

    missing = [c for c in REQUIRED_FEATURES if c not in csv_columns]
    if missing:
        msg = f"Gate 2 FAILED — required feature columns missing from features.csv: {missing}"
        log.error(msg)
        raise RuntimeError(msg)

    log.info("Gate 2 PASSED — all required features present: %s", REQUIRED_FEATURES)


# ── Flow definition ───────────────────────────────────────────────────────────
class AIProductFlow(Flow):
    """End-to-end orchestration: Analyst Crew → Gate 1 → Scientist Crew → Gate 2."""

    def __init__(self):
        super().__init__()
        self.log = _make_logger()

    # ── Step 1: Analyst Crew ─────────────────────────────────────────────────
    @start()
    def run_analyst_crew(self):
        self.log.info("=" * 60)
        self.log.info("STEP 1 — Starting Analyst Crew")
        self.log.info("=" * 60)

        from crew_analyst.crew import run as analyst_run
        try:
            result = analyst_run()
            _check_analyst_outputs(self.log)
            self.log.info("STEP 1 COMPLETE — Analyst Crew finished successfully.")
            return result
        except Exception as exc:
            self.log.error("STEP 1 FAILED: %s", exc)
            raise

    # ── Validation Gate 1 + Step 2: Scientist Crew ──────────────────────────
    @listen(run_analyst_crew)
    def run_scientist_crew(self, analyst_result):
        self.log.info("=" * 60)
        self.log.info("GATE 1 — Validating contract against clean data …")
        self.log.info("=" * 60)
        _gate1_contract_vs_data(self.log)

        self.log.info("=" * 60)
        self.log.info("STEP 2 — Starting Scientist Crew")
        self.log.info("=" * 60)

        from crew_scientist.crew import run as scientist_run
        try:
            result = scientist_run()
            self.log.info("STEP 2 COMPLETE — Scientist Crew finished successfully.")
        except Exception as exc:
            self.log.error("STEP 2 FAILED: %s", exc)
            raise

        self.log.info("=" * 60)
        self.log.info("GATE 2 — Validating feature columns …")
        self.log.info("=" * 60)
        _gate2_features_check(self.log)

        self.log.info("=" * 60)
        self.log.info("FLOW COMPLETE — all steps and gates passed.")
        self.log.info("Outputs: %s", [p.name for p in REQUIRED_ANALYST_OUTPUTS + REQUIRED_SCIENTIST_OUTPUTS])
        self.log.info("=" * 60)
        return result


# ── Entry point ───────────────────────────────────────────────────────────────
def run():
    flow = AIProductFlow()
    return flow.kickoff()


if __name__ == "__main__":
    run()
