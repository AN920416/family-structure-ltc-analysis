"""Comprehensive U-shape relationship analysis without verbose console output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency, fisher_exact

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class PairwiseComparison:
    group: str
    child1: int
    child2: int
    child1_n: int
    child1_users: int
    child1_rate: float
    child2_n: int
    child2_users: int
    child2_rate: float
    chi2: float
    p_value: float
    fisher_p: Optional[float]
    use_fisher: bool
    significant: bool
    contingency_table: List[List[int]]


@dataclass
class LogisticCoefficient:
    label: str
    child_count: Optional[int]
    coefficient: float
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


@dataclass
class LogisticAnalysis:
    group: str
    sample_size: int
    pseudo_r2: float
    coefficients: List[LogisticCoefficient]


@dataclass
class AnalysisOutputs:
    dataset_shape: Tuple[int, int]
    pairwise_overall: List[PairwiseComparison]
    pairwise_stratified: Dict[str, List[PairwiseComparison]]
    logistic_overall: Optional[LogisticAnalysis]
    logistic_stratified: Dict[str, LogisticAnalysis]
    visualization_paths: Dict[str, str]
    derived_metrics: Dict[str, Any]
    stratified_notes: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return _to_native(asdict(self))


NUMERIC_COLUMNS = ["bus", "store", "hospital", "lique"]
COMPARISON_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 3),
    (1, 4),
]
STRATIFIED_KEY_COMPARISONS = [(1, 2), (2, 3), (0, 2)]
KEY_PAIRWISE_COMPARISONS = [(1, 2), (2, 3), (0, 2), (1, 3)]
IMPORTANT_GROUPS = [
    ("75-84", "非低收中低收入戶"),
    ("85-94", "非低收中低收入戶"),
    ("65-74", "非低收中低收入戶"),
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for col in NUMERIC_COLUMNS:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col].str.replace("\"", ""), errors="coerce")

    df["income_group"] = df.apply(_create_income_group, axis=1)
    df["age_group"] = pd.cut(
        df["age"], bins=[64, 74, 84, 94, 120], labels=["65-74", "75-84", "85-94", "95+"], include_lowest=True
    )
    return df


def _create_income_group(row: pd.Series) -> str:
    low_type = row.get("low_type_cd")
    if pd.isna(low_type):
        return "未知"
    if low_type == 99:
        return "非低收中低收入戶"
    if low_type == 5:
        return "中低收入戶"
    if low_type in [0, 1, 2, 3, 4]:
        return "低收入戶"
    return "其他"


def _to_int(value: Any) -> int:
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _to_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def perform_pairwise_chi2_test(
    data: pd.DataFrame, child1: int, child2: int, group_name: str = "整體"
) -> Optional[PairwiseComparison]:
    subset = data[data["child_cnt"].isin([child1, child2])]
    if len(subset) < 10:
        return None

    contingency_table = pd.crosstab(subset["child_cnt"], subset["is_use_long_term_care"])
    if contingency_table.shape != (2, 2):
        return None

    usage_rates = subset.groupby("child_cnt")["is_use_long_term_care"].agg(["count", "sum", "mean"])
    chi2_result = chi2_contingency(contingency_table)
    try:
        chi2_value = float(chi2_result.statistic)  # type: ignore[attr-defined]
        p_value = float(chi2_result.pvalue)  # type: ignore[attr-defined]
    except AttributeError:
        chi2_tuple = cast(Tuple[float, float, float, np.ndarray], chi2_result)
        chi2_value = float(chi2_tuple[0])
        p_value = float(chi2_tuple[1])

    fisher_p: Optional[float]
    if contingency_table.min().min() < 5:
        fisher_result = fisher_exact(contingency_table)
        try:
            fisher_p = float(fisher_result.pvalue)  # type: ignore[attr-defined]
        except AttributeError:
            fisher_tuple = cast(Tuple[float, float], fisher_result)
            fisher_p = float(fisher_tuple[1])
        use_fisher = True
    else:
        fisher_p = None
        use_fisher = False

    significant = bool((fisher_p is not None and fisher_p < 0.05) or (fisher_p is None and p_value < 0.05))

    child1_n = _to_int(usage_rates.at[child1, "count"])
    child1_users = _to_int(usage_rates.at[child1, "sum"])
    child1_rate = _to_float(usage_rates.at[child1, "mean"])
    child2_n = _to_int(usage_rates.at[child2, "count"])
    child2_users = _to_int(usage_rates.at[child2, "sum"])
    child2_rate = _to_float(usage_rates.at[child2, "mean"])

    return PairwiseComparison(
        group=group_name,
        child1=child1,
        child2=child2,
        child1_n=child1_n,
        child1_users=child1_users,
        child1_rate=child1_rate,
        child2_n=child2_n,
        child2_users=child2_users,
        child2_rate=child2_rate,
        chi2=chi2_value,
        p_value=p_value,
        fisher_p=fisher_p,
        use_fisher=bool(use_fisher),
        significant=bool(significant),
        contingency_table=contingency_table.values.astype(int).tolist(),
    )


def compute_pairwise_analysis(
    data: pd.DataFrame, comparison_pairs: Sequence[Tuple[int, int]], group_name: str
) -> List[PairwiseComparison]:
    results: List[PairwiseComparison] = []
    for child1, child2 in comparison_pairs:
        result = perform_pairwise_chi2_test(data, child1, child2, group_name)
        if result:
            results.append(result)
    return results


def compute_stratified_pairwise(
    data: pd.DataFrame,
    groups: Sequence[Tuple[str, str]],
    comparison_pairs: Sequence[Tuple[int, int]],
    min_samples: int = 100,
) -> Tuple[Dict[str, List[PairwiseComparison]], Dict[str, int]]:
    stratified: Dict[str, List[PairwiseComparison]] = {}
    skipped: Dict[str, int] = {}

    for age_group, income_group in groups:
        subset = data[(data["age_group"] == age_group) & (data["income_group"] == income_group)]
        key = f"{age_group}×{income_group}"
        if len(subset) < min_samples:
            skipped[key] = len(subset)
            continue
        group_results = compute_pairwise_analysis(subset, comparison_pairs, key)
        if group_results:
            stratified[key] = group_results
    return stratified, skipped


def perform_logistic_regression_analysis(
    data: pd.DataFrame, group_name: str = "整體", max_children: int = 8, min_samples: int = 100
) -> Optional[LogisticAnalysis]:
    analysis_data = data[data["child_cnt"] <= max_children].copy()
    if len(analysis_data) < min_samples:
        return None

    analysis_data["child_cnt_cat"] = analysis_data["child_cnt"].astype(str)
    child_dummies = pd.get_dummies(analysis_data["child_cnt_cat"], prefix="child", drop_first=False)
    if "child_0" in child_dummies.columns:
        child_dummies = child_dummies.drop("child_0", axis=1)

    X = sm.add_constant(child_dummies.astype(float))
    y = analysis_data["is_use_long_term_care"].astype(float)

    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=0)
    except Exception:
        return None

    conf_int = result.conf_int()
    coefficients = result.params
    p_values = result.pvalues
    odds_ratios = np.exp(coefficients)

    coeffs: List[LogisticCoefficient] = []
    for label, coef in coefficients.items():
        ci_lower, ci_upper = conf_int.loc[label]
        coeffs.append(
            LogisticCoefficient(
                label=label,
                child_count=int(label.replace("child_", "")) if label.startswith("child_") else None,
                coefficient=float(coef),
                odds_ratio=float(np.exp(coef)),
                ci_lower=float(np.exp(ci_lower)),
                ci_upper=float(np.exp(ci_upper)),
                p_value=float(p_values[label]),
                significant=bool(p_values[label] < 0.05),
            )
        )

    return LogisticAnalysis(
        group=group_name,
        sample_size=int(len(analysis_data)),
        pseudo_r2=float(result.prsquared),
        coefficients=coeffs,
    )


def compute_stratified_logistic(
    data: pd.DataFrame,
    groups: Sequence[Tuple[str, str]],
    min_samples: int = 100,
) -> Tuple[Dict[str, LogisticAnalysis], Dict[str, int]]:
    stratified: Dict[str, LogisticAnalysis] = {}
    skipped: Dict[str, int] = {}

    for age_group, income_group in groups:
        subset = data[(data["age_group"] == age_group) & (data["income_group"] == income_group)]
        key = f"{age_group}×{income_group}"
        result = perform_logistic_regression_analysis(subset, key, min_samples=min_samples)
        if result:
            stratified[key] = result
        else:
            skipped[key] = len(subset)
    return stratified, skipped


def create_visualizations(
    df: pd.DataFrame,
    pairwise_results: Sequence[PairwiseComparison],
    logistic_result: Optional[LogisticAnalysis],
    output_path: str,
) -> str:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "U-Shape Statistical Validation: Pairwise & Logistic Analysis",
        fontsize=14,
        y=0.98,
    )

    if pairwise_results:
        max_child = max(max(r.child1, r.child2) for r in pairwise_results)
        size = max_child + 1
        p_matrix = np.full((size, size), np.nan)
        for result in pairwise_results:
            p_val = result.fisher_p if result.fisher_p is not None else result.p_value
            p_matrix[result.child1, result.child2] = p_val
            p_matrix[result.child2, result.child1] = p_val
        np.fill_diagonal(p_matrix, np.nan)
        mask = np.isnan(p_matrix)
        sns.heatmap(
            p_matrix,
            ax=axes[0, 0],
            cmap="RdYlBu_r",
            vmin=0,
            vmax=0.05,
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "p-value"},
            mask=mask,
            square=True,
        )
        axes[0, 0].set_title("Pairwise p-value matrix")
        axes[0, 0].set_xlabel("Number of children")
        axes[0, 0].set_ylabel("Number of children")
    else:
        axes[0, 0].axis("off")

    if logistic_result:
        child_coeffs = [c for c in logistic_result.coefficients if c.child_count is not None]
        child_coeffs = sorted(child_coeffs, key=lambda c: cast(int, c.child_count))
        if child_coeffs:
            xs = [cast(int, c.child_count) for c in child_coeffs]
            ys = [c.coefficient for c in child_coeffs]
            colors = ["red" if c.significant else "blue" for c in child_coeffs]
            axes[0, 1].bar(xs, ys, color=colors, alpha=0.7)
            axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[0, 1].set_title("Logistic coefficients (vs 0 children)")
            axes[0, 1].set_xlabel("Number of children")
            axes[0, 1].set_ylabel("Coefficient")
            axes[0, 1].grid(True, alpha=0.3)
            for child, coef, coeff_obj in zip(xs, ys, child_coeffs):
                if coeff_obj.significant:
                    axes[0, 1].text(child, coef + 0.01, "*", ha="center", va="bottom", color="white", fontsize=12)
        else:
            axes[0, 1].axis("off")
    else:
        axes[0, 1].axis("off")

    usage = df.groupby("child_cnt")["is_use_long_term_care"].agg(["count", "mean"]).reset_index()
    usage = usage[usage["count"] >= 100]
    if not usage.empty:
        axes[1, 0].plot(usage["child_cnt"], usage["mean"] * 100, "o-", linewidth=2, markersize=6)
        axes[1, 0].set_title("Long-term care usage trend")
        axes[1, 0].set_xlabel("Number of children")
        axes[1, 0].set_ylabel("Usage rate (%)")
        axes[1, 0].grid(True, alpha=0.3)
        for _, row in usage.iterrows():
            axes[1, 0].annotate(
                f"n={int(row['count'])}",
                (row["child_cnt"], row["mean"] * 100),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    else:
        axes[1, 0].axis("off")

    if pairwise_results:
        comparison_types = {
            "Adjacent pairs": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            "Skip-one pairs": [(0, 2), (1, 3), (2, 4)],
            "Skip-two pairs": [(0, 3), (1, 4)],
        }
        type_results = {}
        for comp_type, pairs in comparison_types.items():
            totals = [
                result
                for pair in pairs
                for result in pairwise_results
                if {result.child1, result.child2} == set(pair)
            ]
            if totals:
                sig_pct = sum(r.significant for r in totals) / len(totals) * 100
                type_results[comp_type] = sig_pct
        if type_results:
            types = list(type_results.keys())
            percentages = list(type_results.values())
            bars = axes[1, 1].bar(types, percentages, color=["skyblue", "lightgreen", "lightcoral"])
            axes[1, 1].set_title("Share of significant comparisons")
            axes[1, 1].set_ylabel("Share (%)")
            max_pct = max(percentages)
            axes[1, 1].set_ylim(0, max(110, max_pct * 1.1))
            for bar, pct in zip(bars, percentages):
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 5,
                    f"{pct:.1f}%",
                    ha="center",
                    va="top",
                    color="white" if pct > 20 else "black",
                    fontsize=10,
                )
        else:
            axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def summarize_pairwise(results: Sequence[PairwiseComparison]) -> Dict[str, Any]:
    total = len(results)
    significant = sum(r.significant for r in results)
    key_details = {}
    for child1, child2 in KEY_PAIRWISE_COMPARISONS:
        match = next((r for r in results if r.child1 == child1 and r.child2 == child2), None)
        if match:
            direction = "up" if match.child2_rate > match.child1_rate else "down"
            p_val = match.fisher_p if match.fisher_p is not None else match.p_value
            key_details[f"{child1}_vs_{child2}"] = {
                "child1_rate": match.child1_rate,
                "child2_rate": match.child2_rate,
                "p_value": p_val,
                "significant": match.significant,
                "direction": direction,
            }
    return {
        "total_comparisons": total,
        "significant_comparisons": significant,
        "significant_ratio": (significant / total) if total else 0,
        "key_comparisons": key_details,
    }


def summarize_logistic(result: Optional[LogisticAnalysis]) -> Dict[str, Any]:
    if not result:
        return {}
    child_coeffs = [c for c in result.coefficients if c.child_count is not None]
    if not child_coeffs:
        return {
            "sample_size": result.sample_size,
            "pseudo_r2": result.pseudo_r2,
            "coefficients": [],
        }
    min_coeff = min(child_coeffs, key=lambda c: c.coefficient)
    u_shape_detected = 0 < child_coeffs.index(min_coeff) < len(child_coeffs) - 1
    return {
        "sample_size": result.sample_size,
        "pseudo_r2": result.pseudo_r2,
        "min_risk_child": min_coeff.child_count,
        "min_risk_coefficient": min_coeff.coefficient,
        "u_shape_pattern": u_shape_detected,
        "coefficients": [asdict(c) for c in child_coeffs],
    }


def build_derived_metrics(
    pairwise_overall: Sequence[PairwiseComparison],
    logistic_overall: Optional[LogisticAnalysis],
    pairwise_stratified: Dict[str, List[PairwiseComparison]],
    logistic_stratified: Dict[str, LogisticAnalysis],
) -> Dict[str, Any]:
    return {
        "pairwise_summary": summarize_pairwise(pairwise_overall),
        "logistic_summary": summarize_logistic(logistic_overall),
        "stratified_pairwise_counts": {k: len(v) for k, v in pairwise_stratified.items()},
        "stratified_logistic_counts": {k: len(v.coefficients) for k, v in logistic_stratified.items()},
    }


def save_summary(payload: Dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def run_analysis(
    csv_path: str = "data/dataset/angels.csv",
    figure_path: str = "visualizations/u_shape_analysis.png",
    summary_path: str = "results/u_shape_analysis_summary.json",
) -> AnalysisOutputs:
    df = load_dataset(csv_path)
    pairwise_overall = compute_pairwise_analysis(df, COMPARISON_PAIRS, "整體")
    stratified_pairwise, pairwise_skipped = compute_stratified_pairwise(df, IMPORTANT_GROUPS, STRATIFIED_KEY_COMPARISONS)
    logistic_overall = perform_logistic_regression_analysis(df, "整體")
    stratified_logistic, logistic_skipped = compute_stratified_logistic(df, IMPORTANT_GROUPS)

    visualization_path = create_visualizations(df, pairwise_overall, logistic_overall, figure_path)
    derived_metrics = build_derived_metrics(
        pairwise_overall, logistic_overall, stratified_pairwise, stratified_logistic
    )

    outputs = AnalysisOutputs(
        dataset_shape=df.shape,
        pairwise_overall=pairwise_overall,
        pairwise_stratified=stratified_pairwise,
        logistic_overall=logistic_overall,
        logistic_stratified=stratified_logistic,
        visualization_paths={"u_shape_visualization": visualization_path},
        derived_metrics=derived_metrics,
        stratified_notes={
            "pairwise_skipped": pairwise_skipped,
            "logistic_skipped": logistic_skipped,
        },
    )

    if summary_path:
        save_summary(outputs.to_dict(), summary_path)

    return outputs


def _to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


if __name__ == "__main__":
    run_analysis()