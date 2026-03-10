"""
Data Analyst Crew — Agent definitions.

Uses OpenAI gpt-4o-mini for LLM reasoning. All real data work is done
inside the @tool functions; result_as_answer=True returns the tool output
directly as the task answer, minimising token usage.
"""

from crewai import Agent
from crew_analyst.tools import (
    validate_raw_data,
    clean_data,
    run_eda_report,
    generate_dataset_contract,
)


def get_data_ingestor() -> Agent:
    return Agent(
        role="Data Ingestor and Validator",
        goal=(
            "Load the raw online_retail_II.csv dataset, verify its schema, "
            "count nulls and duplicates, and produce a validation report."
        ),
        backstory=(
            "An experienced data engineer who catalogues every data quality "
            "issue before the dataset is used downstream."
        ),
        tools=[validate_raw_data],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
    )


def get_data_cleaner() -> Agent:
    return Agent(
        role="Data Cleaner",
        goal=(
            "Transform the raw dataset into a clean, analysis-ready CSV by "
            "removing duplicates, cancellations, and invalid rows, then "
            "fixing data types and saving outputs/clean_data.csv."
        ),
        backstory=(
            "A meticulous data engineer who applies zero-tolerance cleaning "
            "rules and documents every row removed."
        ),
        tools=[clean_data],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
    )


def get_eda_analyst() -> Agent:
    return Agent(
        role="EDA and Insights Analyst",
        goal=(
            "Run exploratory data analysis on the cleaned dataset, generate "
            "an HTML EDA report, derive at least 5 actionable business insights, "
            "and save them to outputs/insights.md."
        ),
        backstory=(
            "A senior retail data analyst who translates raw statistics into "
            "clear, actionable business findings."
        ),
        tools=[run_eda_report],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
        cache=False,
    )


def get_contract_author() -> Agent:
    return Agent(
        role="Dataset Contract Author",
        goal=(
            "Produce outputs/dataset_contract.json documenting the schema, "
            "constraints, and data quality guarantees of the cleaned dataset."
        ),
        backstory=(
            "A data governance specialist who writes formal data contracts "
            "used by downstream ML engineers and analysts."
        ),
        tools=[generate_dataset_contract],
        llm="gpt-4o-mini",
        verbose=True,
        allow_delegation=False,
    )
