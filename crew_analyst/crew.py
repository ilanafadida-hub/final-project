"""
Data Analyst Crew — Crew assembly and entry point.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path and load env FIRST
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")  # load ANTHROPIC_API_KEY before any crewai import

from crewai import Crew, Process
from crew_analyst.tasks import get_tasks


def run():
    tasks, agents = get_tasks()

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Starting Data Analyst Crew...")
    print("=" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Crew finished. Verifying output files...")
    print("=" * 60)

    outputs_dir = _project_root / "outputs"
    expected = [
        "clean_data.csv",
        "eda_report.html",
        "insights.md",
        "dataset_contract.json",
    ]
    all_ok = True
    for fname in expected:
        fpath = outputs_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  [OK] {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    if all_ok:
        print("\nAll output files produced successfully.")
    else:
        print("\nSome output files are missing — check logs above.")

    return result


if __name__ == "__main__":
    run()
