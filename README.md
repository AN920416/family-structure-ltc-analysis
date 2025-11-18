# Elder Care Analysis · Angels Dataset

## Project Overview
This repository hosts a reproducible analysis pipeline built around Taiwan's Angels Dataset (420,510 older adults). The goal is to understand how household composition, income security, and neighborhood amenities influence long-term-care (LTC) enrollment, with a particular focus on the U-shaped relationship between number of children and service uptake. Findings are packaged as machine-readable summaries and English-only visualizations to support evidence-based policy discussions.

## Repository Structure
```
analysis/                # Main analysis entry point (analysis.py)
archive/                 # Legacy notebooks, scripts, and historical outputs
data/dataset/angels.csv  # Raw dataset (not versioned; expected locally)
docs/                    # Background notes and report drafts
results/                 # Generated tables & JSON summaries (tracked)
visualizations/          # Exported charts (English labels)
venv/                    # Optional local virtual environment
```

## Analysis Workflow
### `analysis/analysis.py`
The primary script orchestrates the following stages:
1. **Data loading & cleaning** – standardizes numeric columns, buckets ages, and translates income codes into human-readable groups.
2. **Pairwise statistical tests** – runs chi-square/Fisher exact tests across curated child-count comparisons to validate the U-shape hypothesis at both population and stratified (age × income) levels.
3. **Logistic regression** – estimates odds ratios for LTC usage by child count, including stratified specifications for high-priority cohorts.
4. **Visualization** – produces an English heatmap of pairwise p-values, logistic coefficients, usage trends, and comparison summaries (saved as `visualizations/u_shape_analysis.png`).
5. **Structured outputs** – writes consolidated metrics, coefficients, and metadata to `results/u_shape_analysis_summary.json` without flooding the console.

### Archive Materials
Older exploratory analyses (EDA, vulnerability indices, and LTC personas) plus their Markdown/CSV artifacts live under `archive/`. They illustrate the project evolution and may be referenced for extended reporting but are not part of the default run.

## Key Insights (current run)
- LTC participation remains low (~2.9%) despite sizable eligible populations, underscoring capacity or awareness gaps.
- Usage rates decline sharply among elders with exactly one child, rise for 2–3 children, and taper again at higher counts, forming the validated U-shape.
- Stratified tests show the effect is strongest for ages 75–94 in non-low-income households, while low-income cohorts display flatter curves due to universally high needs.
- Access variables (bus/store/hospital indices) correlate with uptake only when child counts exceed two, suggesting family support moderates environmental constraints.

## Getting Started
1. **Python environment** – Python 3.10+ is recommended. Either reuse `venv/` or create your own virtual environment.
2. **Install dependencies** (if a requirements file is absent):
	```bash
	pip install pandas numpy seaborn matplotlib scipy statsmodels
	```
3. **Place the dataset** at `data/dataset/angels.csv`. The file is large and excluded from git; obtain it through the authorized data-sharing process.

## Running the Pipeline
```bash
python analysis/analysis.py
```
The script automatically regenerates the visualization and JSON summary under `visualizations/` and `results/`. Existing files are overwritten, so archive previous runs if needed.

## Outputs
- `visualizations/u_shape_analysis.png` – heatmap & trend plot validating the U-shaped relationship.
- `results/u_shape_analysis_summary.json` – machine-readable snapshot containing dataset stats, pairwise tests, logistic coefficients, and derived metrics.

## Extending the Work
- Integrate covariates such as disability status or living arrangement into the logistic stage to explain residual variance.
- Parameterize child-count buckets or stratification schemes via CLI flags to support rapid what-if simulations.
- Convert the archive notebooks into formal reports housed in `docs/` for publication-ready deliverables.

## License & Attribution
This repository analyzes government-provided Angels data; ensure compliance with the source data license and anonymization rules when sharing derived results. Analytical code is provided for research and policy evaluation purposes.