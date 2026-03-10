"""
Data Scientist Crew — Task definitions (sequential).
"""

from crewai import Task
from crew_scientist.agents import (
    get_contract_validator,
    get_feature_engineer,
    get_model_trainer,
    get_model_card_author,
)
from crew_scientist.tools import (
    validate_contract,
    engineer_features,
    train_models,
    save_evaluation_report,
    save_model_card,
)


def get_tasks():
    validator = get_contract_validator()
    engineer = get_feature_engineer()
    trainer = get_model_trainer()
    card_author = get_model_card_author()

    task1 = Task(
        name="validate_contract_task",
        description=(
            "Call validate_contract() to load outputs/clean_data.csv and "
            "outputs/dataset_contract.json, verify that all required columns "
            "(Invoice, StockCode, Description, Quantity, InvoiceDate, Price, "
            "Customer ID, Country) are present with the expected dtypes, and "
            "return a pass/fail validation report."
        ),
        expected_output=(
            "A validation report confirming all required columns are present "
            "with correct dtypes, and an overall PASS or FAIL result."
        ),
        agent=validator,
        tools=[validate_contract],
    )

    task2 = Task(
        name="engineer_features_task",
        description=(
            "Call engineer_features() to compute RFM features per customer from "
            "outputs/clean_data.csv:\n"
            "  - Recency: days since last purchase (reference = max InvoiceDate)\n"
            "  - Frequency: number of unique invoices per customer\n"
            "  - Monetary: total spend (Quantity * Price) per customer\n"
            "  - AvgOrderValue: Monetary / Frequency\n"
            "  - UniqueProducts: distinct StockCodes per customer\n"
            "Scale all features with StandardScaler and save to "
            "outputs/features.csv (one row per Customer ID)."
        ),
        expected_output=(
            "Confirmation that outputs/features.csv was saved with one row per "
            "customer and 5 standardised RFM features, plus a summary of "
            "customer count and feature statistics."
        ),
        agent=engineer,
        tools=[engineer_features],
        context=[task1],
    )

    task3 = Task(
        name="train_and_evaluate_task",
        description=(
            "Call train_models() to load outputs/features.csv, train two models:\n"
            "  1. KMeans (n_clusters=4, random_state=42)\n"
            "  2. AgglomerativeClustering (n_clusters=4)\n"
            "Compute silhouette scores for both and inertia for KMeans. "
            "Select the model with the higher silhouette score as the best model. "
            "Save the best model to outputs/model.pkl using joblib. "
            "Save a markdown evaluation report to outputs/evaluation_report.md "
            "including the metrics table and cluster size distributions. "
            "Return the full metrics comparison string."
        ),
        expected_output=(
            "A metrics comparison showing silhouette scores for both models, "
            "the name of the best model, confirmation that outputs/model.pkl "
            "was saved, and confirmation that outputs/evaluation_report.md was saved."
        ),
        agent=trainer,
        tools=[train_models, save_evaluation_report],
        context=[task2],
    )

    task4 = Task(
        name="write_model_card_task",
        description=(
            "Call save_model_card() to generate and save a comprehensive model card "
            "to outputs/model_card.md. The model card must include:\n"
            "  - Model overview (task, algorithm, features, preprocessing)\n"
            "  - Intended use and out-of-scope uses\n"
            "  - Training data description\n"
            "  - Feature descriptions\n"
            "  - Performance metrics (from the evaluation report)\n"
            "  - Ethical considerations\n"
            "  - Limitations\n"
            "  - Usage example (Python code)\n"
        ),
        expected_output=(
            "Confirmation that outputs/model_card.md was saved with all required "
            "sections documented."
        ),
        agent=card_author,
        tools=[save_model_card],
        context=[task3],
    )

    return [task1, task2, task3, task4], [validator, engineer, trainer, card_author]
