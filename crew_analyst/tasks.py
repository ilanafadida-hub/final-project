"""
Data Analyst Crew — Task definitions (sequential).
"""

from crewai import Task
from crew_analyst.agents import (
    get_data_ingestor,
    get_data_cleaner,
    get_eda_analyst,
    get_contract_author,
)
from crew_analyst.tools import (
    validate_raw_data,
    clean_data,
    run_eda_report,
    save_insights,
    generate_dataset_contract,
)


def get_tasks():
    ingestor = get_data_ingestor()
    cleaner = get_data_cleaner()
    analyst = get_eda_analyst()
    contract_author = get_contract_author()

    task1 = Task(
        name="validate_raw_data_task",
        description=(
            "Call the validate_raw_data tool to load data/raw/online_retail_II.csv "
            "and produce a full validation report. "
            "The report must include: total row/column counts, which columns are present, "
            "null counts per column, number of duplicate rows, "
            "number of cancellation invoices (starting with 'C'), "
            "and number of rows with non-positive Quantity. "
            "Return the full report text."
        ),
        expected_output=(
            "A detailed validation report string covering schema check, null counts, "
            "duplicate count, cancellation count, and negative quantity count."
        ),
        agent=ingestor,
        tools=[validate_raw_data],
    )

    task2 = Task(
        name="clean_data_task",
        description=(
            "Call the clean_data tool to clean the raw dataset. "
            "The tool will: remove duplicate rows, remove cancellation invoices "
            "(Invoice starting with 'C'), remove rows where Quantity <= 0, "
            "remove rows where Price <= 0, drop rows with missing Customer ID, "
            "fix InvoiceDate to datetime and Customer ID to integer, "
            "and save the result to outputs/clean_data.csv. "
            "Return the cleaning summary."
        ),
        expected_output=(
            "A cleaning summary showing how many rows were removed at each step "
            "and confirming that outputs/clean_data.csv was saved."
        ),
        agent=cleaner,
        tools=[clean_data],
        context=[task1],
    )

    task3 = Task(
        name="eda_and_insights_task",
        description=(
            "Perform EDA and write business insights. Follow these steps:\n\n"
            "STEP 1: Call run_eda_report() to generate the EDA HTML report and get "
            "the key statistics string (top products, countries, monthly revenue, "
            "customer spend, etc.).\n\n"
            "STEP 2: Using the statistics returned by run_eda_report(), reason as a "
            "senior business analyst and write AT LEAST 5 clear, actionable business insights "
            "in markdown format. Each insight should:\n"
            "  - Have a numbered heading (## 1. Title)\n"
            "  - Cite specific numbers from the stats\n"
            "  - Explain what it means for the business\n"
            "  - Suggest a possible action\n\n"
            "STEP 3: Call save_insights(insights_text) with the full markdown text you wrote.\n\n"
            "The markdown must begin with: '# Business Insights — Online Retail II'\n"
            "and include a 'Generated: <date>' line, then at least 5 numbered insight sections."
        ),
        expected_output=(
            "Confirmation that the EDA HTML report was saved to outputs/eda_report.html "
            "and that at least 5 business insights were written and saved to outputs/insights.md."
        ),
        agent=analyst,
        tools=[run_eda_report, save_insights],
        context=[task2],
    )

    task4 = Task(
        name="generate_contract_task",
        description=(
            "Call generate_dataset_contract() to read outputs/clean_data.csv, "
            "infer the schema for each column (dtype, nullable flag, null rate, "
            "min/max values, sample allowed values), and save outputs/dataset_contract.json. "
            "The contract must include: version, created_at timestamp, row_count, "
            "column_count, per-column schema, and a list of data quality constraints. "
            "Return the contract summary."
        ),
        expected_output=(
            "Confirmation that outputs/dataset_contract.json was saved, "
            "with a summary of the schema and constraints documented."
        ),
        agent=contract_author,
        tools=[generate_dataset_contract],
        context=[task2],
    )

    return [task1, task2, task3, task4], [ingestor, cleaner, analyst, contract_author]
