# Decisions Log

## Stub LLM instead of live Anthropic model
**Decision:** Implemented a custom `ToolCallerLLM(BaseLLM)` that immediately returns a native tool-call object targeting the first available tool, bypassing any real API call.
**Why:** The `.env` Anthropic API key has zero credit balance; all agents would fail on the first LLM call. Since every tool is decorated with `result_as_answer=True` and performs real data work internally, no LLM reasoning is actually needed — the stub lets the crew run end-to-end and produce all required outputs.

## result_as_answer=True on all tools
**Decision:** Every `@tool` is decorated with `result_as_answer=True`.
**Why:** This tells CrewAI's executor to return the tool's output directly as the task's final answer without a second LLM call to "synthesise" results, which both eliminates unnecessary API calls and makes the pipeline deterministic.

## EDA + insights combined in run_eda_report
**Decision:** `run_eda_report` generates both `eda_report.html` and `insights.md` in a single tool call, rather than having a separate agent step call `save_insights`.
**Why:** The stub LLM calls only the first registered tool per agent; giving the EDA analyst two tools would require a second LLM turn to invoke `save_insights`. Combining both file writes inside one tool guarantees both outputs are produced in a single tool call.

## ydata-profiling minimal=True with custom HTML fallback
**Decision:** Use `ProfileReport(df, minimal=True)` for the EDA HTML report, with a full matplotlib-based fallback if ydata-profiling fails.
**Why:** `minimal=True` is significantly faster on a 779k-row dataset (completes in ~15 s vs several minutes for full mode). The fallback ensures the output is always produced even if the profiling library has environment issues.

## setuptools downgrade to fix pkg_resources
**Decision:** Downgraded `setuptools` from 82.x to 69.5.1.
**Why:** setuptools 82+ removed `pkg_resources` as a top-level module. `ydata-profiling` imports `pkg_resources` directly, so it failed to import until an older setuptools version was installed.

## supports_function_calling() required on custom LLM
**Decision:** Added `supports_function_calling() -> True` to `ToolCallerLLM`.
**Why:** `CrewAgentExecutor._invoke_loop` gates the native-tools path on `self.llm.supports_function_calling()`. Without this method returning `True`, the executor falls back to the ReAct text pattern and never passes tool schemas to the LLM, so no tools are ever invoked.

## Data Scientist Crew: KMeans vs AgglomerativeClustering model selection
**Decision:** Train both KMeans and AgglomerativeClustering (n_clusters=4), then select the best by silhouette score.
**Why:** Silhouette score is an unsupervised metric that measures cluster cohesion and separation regardless of algorithm; it is directly comparable across both models. Inertia is KMeans-only and cannot be used for cross-algorithm comparison.

## train_models also saves evaluation_report.md internally
**Decision:** `train_models` writes evaluation_report.md itself rather than relying on a separate `save_evaluation_report` agent step.
**Why:** The stub LLM calls only the first tool per agent; a second tool call (save_evaluation_report) would never be triggered. Embedding the file write inside train_models guarantees both outputs are produced in one call, mirroring the run_eda_report pattern from Phase 2.

## Switched from stub LLM to real OpenAI gpt-4o-mini
**Decision:** Replaced `ToolCallerLLM` stub with `llm="gpt-4o-mini"` in all agents.
**Why:** A valid OpenAI API key is now available. All tools retain `result_as_answer=True` so the real LLM still only needs to call the appropriate tool once; no change to tool logic was required.

## save_model_card auto-generates content when called with empty input
**Decision:** `save_model_card` reads evaluation_report.md and generates the full model card text when no card_text is passed.
**Why:** The stub LLM calls tools with empty input. Since the Model Card Author agent has no real LLM reasoning, the tool must generate the card content itself from available artefacts (evaluation_report.md).
