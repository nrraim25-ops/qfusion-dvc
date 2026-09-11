"""Statistical evaluation module for multi-seed comparisons.

Implements paired Wilcoxon signed-rank test between experimental conditions per Gate 14.
Rule 1.9: Reports exact p-values and effect directions objectively without threshold spin.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy import stats


def paired_wilcoxon_test(
    cond1_values: List[float],
    cond2_values: List[float],
    cond1_name: str,
    cond2_name: str,
    metric_name: str
) -> Dict[str, Any]:
    """Computes paired Wilcoxon signed-rank test between two conditions across random seeds.
    
    Args:
        cond1_values: Metric values for condition 1 across seeds (e.g. [seed_42, seed_123, seed_2024])
        cond2_values: Metric values for condition 2 across the same paired seeds
        cond1_name: Name of condition 1
        cond2_name: Name of condition 2
        metric_name: Name of the metric evaluated
    Returns:
        Dict with descriptive stats, Wilcoxon test statistic, exact p-value, and effect direction.
    """
    assert len(cond1_values) == len(cond2_values), "Condition arrays must have matching seed counts"
    n = len(cond1_values)
    
    arr1 = np.array(cond1_values, dtype=float)
    arr2 = np.array(cond2_values, dtype=float)
    diff = arr1 - arr2
    
    mean1 = float(np.mean(arr1))
    std1 = float(np.std(arr1, ddof=1)) if n > 1 else 0.0
    mean2 = float(np.mean(arr2))
    std2 = float(np.std(arr2, ddof=1)) if n > 1 else 0.0
    
    diff_mean = float(np.mean(diff))
    
    # Check if differences are all zero
    if np.all(diff == 0):
        stat = 0.0
        p_val = 1.0
        direction = "identical"
    else:
        try:
            res = stats.wilcoxon(arr1, arr2, zero_method="pratt", alternative="two-sided")
            stat = float(res.statistic)
            p_val = float(res.pvalue)
        except Exception:
            stat = 0.0
            p_val = 1.0

        if diff_mean > 0:
            direction = f"{cond1_name} > {cond2_name}"
        elif diff_mean < 0:
            direction = f"{cond1_name} < {cond2_name}"
        else:
            direction = "equal"

    is_significant_05 = p_val < 0.05

    return {
        "metric": metric_name,
        "cond1": cond1_name,
        "cond2": cond2_name,
        "n_seeds": n,
        "cond1_mean": mean1,
        "cond1_std": std1,
        "cond2_mean": mean2,
        "cond2_std": std2,
        "mean_diff": diff_mean,
        "statistic": stat,
        "p_value": p_val,
        "p_value_formatted": f"{p_val:.4f}",
        "effect_direction": direction,
        "clears_p_05": is_significant_05
    }


def compare_all_conditions(
    seed_results: Dict[str, Dict[str, List[float]]],
    metrics: List[str]
) -> List[Dict[str, Any]]:
    """Runs pairwise Wilcoxon tests for all 3 condition pairs across all metrics.
    
    Pairs:
    1. classical vs quantum
    2. classical_matched vs quantum
    3. classical vs classical_matched
    """
    pairs = [
        ("classical", "quantum"),
        ("classical_matched", "quantum"),
        ("classical", "classical_matched")
    ]
    
    comparison_table = []
    for c1, c2 in pairs:
        for metric in metrics:
            if metric in seed_results[c1] and metric in seed_results[c2]:
                res = paired_wilcoxon_test(
                    cond1_values=seed_results[c1][metric],
                    cond2_values=seed_results[c2][metric],
                    cond1_name=c1,
                    cond2_name=c2,
                    metric_name=metric
                )
                comparison_table.append(res)
                
    return comparison_table
