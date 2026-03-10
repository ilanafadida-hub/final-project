# Decisions Log

## LLM choice: OpenAI gpt-4o-mini
**Decision:** Used `llm="gpt-4o-mini"` in all agents.
**Why:** A valid OpenAI API key was available and gpt-4o-mini provides a good balance of capability and cost for tool-calling pipelines.

## result_as_answer=True on all tools
**Decision:** Every `@tool` is decorated with `result_as_answer=True`.
**Why:** This tells CrewAI's executor to return the tool's output directly as the task's final answer without a second LLM call to synthesise results, making the pipeline deterministic and eliminating unnecessary API calls.

## ydata-profiling minimal=True with custom HTML fallback
**Decision:** Use `ProfileReport(df, minimal=True)` for the EDA HTML report, with a full matplotlib-based fallback if ydata-profiling fails.
**Why:** `minimal=True` is significantly faster on a 779k-row dataset (~15 s vs several minutes for full mode). The fallback ensures output is always produced even if the profiling library has environment issues.

## Model selection: KMeans vs AgglomerativeClustering
**Decision:** Train both KMeans and AgglomerativeClustering (n_clusters=4), then select the best by silhouette score.
**Why:** Silhouette score is an unsupervised metric that measures cluster cohesion and separation regardless of algorithm, making it directly comparable across both models. Inertia is KMeans-only and cannot be used for cross-algorithm comparison.
