"""
Data Scientist Crew — Agent definitions.

Uses OpenAI gpt-4o-mini for LLM reasoning. All real data / ML work is done
inside the @tool functions; result_as_answer=True returns the tool output
directly as the task answer, minimising token usage.
"""

from crewai import Agent
from crew_scientist.tools import (
    validate_contract,
    engineer_features,
    train_models,
    save_evaluation_report,
    save_model_card,
)


def get_contract_validator() -> Agent:
    return Agent(
        role="Contract Validator",
        goal=(
            "Validate that outputs/clean_data.csv conforms to the schema and "
            "constraints defined in outputs/dataset_contract.json. "
            "Produce a pass/fail report."
        ),
        backstory=(
            "A data governance engineer who verifies every dataset against its "
            "formal contract before any ML work begins."
        ),
        tools=[validate_contract],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
    )


def get_feature_engineer() -> Agent:
    return Agent(
        role="Feature Engineer",
        goal=(
            "Compute RFM features (Recency, Frequency, Monetary, AvgOrderValue, "
            "UniqueProducts) per customer from clean_data.csv, scale them with "
            "StandardScaler, and save outputs/features.csv."
        ),
        backstory=(
            "A machine learning engineer who transforms raw transactional data "
            "into clean, standardised feature vectors ready for clustering."
        ),
        tools=[engineer_features],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
    )


def get_model_trainer() -> Agent:
    return Agent(
        role="Model Trainer and Evaluator",
        goal=(
            "Train KMeans and AgglomerativeClustering (both n_clusters=4) on "
            "outputs/features.csv, compare with silhouette score, save the best "
            "model as outputs/model.pkl, and save outputs/evaluation_report.md."
        ),
        backstory=(
            "An ML engineer who trains, evaluates, and selects the best clustering "
            "model using rigorous quantitative metrics."
        ),
        tools=[train_models, save_evaluation_report],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
        cache=False,
    )


def get_model_card_author() -> Agent:
    return Agent(
        role="Model Card Author",
        goal=(
            "Write a comprehensive model card for the customer segmentation model "
            "and save it to outputs/model_card.md. The card must document the model "
            "purpose, training data, features, performance, limitations, and usage."
        ),
        backstory=(
            "A responsible AI practitioner who documents ML models clearly so "
            "stakeholders can understand capabilities, limitations, and ethical "
            "considerations."
        ),
        tools=[save_model_card],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
    )
