from comp_budgets import get_result_list_trove, get_result_list_primitive
from collections import defaultdict
from pathlib import Path
import numpy as np

def get_solution_diversity(root_dir, trove_or_primitive):
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]

    diversity_per_mode = defaultdict(dict)

    if trove_or_primitive == "trove":
        eval_func = get_result_list_trove
    elif trove_or_primitive == "primitive":
        eval_func = get_result_list_primitive
    else:
        raise ValueError(f"Unknown trove_or_primitive: {trove_or_primitive}")

    for trove in trove_dirs:
        mean_lengths = []
        for updated_md in trove.rglob("*.md"):
            if "backup" in updated_md.parts:
                continue
            means = eval_func(str(updated_md))
            mean_lengths.append(means)

        if mean_lengths:
            diversity_per_mode[str(trove.name)] = {
                "mean": np.mean(mean_lengths),
                "std": np.std(mean_lengths),
            }

    return diversity_per_mode