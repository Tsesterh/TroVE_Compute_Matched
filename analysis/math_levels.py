from comp_budgets import evaluate_markdown_trove, evaluate_markdown_trove_oracle, evaluate_markdown_primitive, evaluate_markdown_primitive_oracle
from collections import defaultdict
from pathlib import Path
from collections import Counter
import json


def get_n_solutions(root_dir, trove_or_prim, max_answers=1):
    """
    Walk `root_dir`, return {trove_dir: {run_name: solved_set}}.
    Only considers *.selected.json* (normal runs).
    """
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]
    
    solution_indices = defaultdict(list)


    for trove in trove_dirs:
        all_correct_indices = []
        accuracies = []
        for updated_md in trove.rglob("*.md"):
            if "backup" in updated_md.parts:          # <─ key line
                continue
            #print(str(updated_md))

            if trove_or_prim == "trove":
                eval_func = evaluate_markdown_trove
            elif trove_or_prim == "trove_oracle":
                eval_func = evaluate_markdown_trove_oracle
            elif trove_or_prim == "primitive":
                eval_func = evaluate_markdown_primitive
            elif trove_or_prim == "primitive_oracle":
                    eval_func = evaluate_markdown_primitive_oracle
            else:
                raise ValueError(f"Unknown trove_or_prim: {trove_or_prim}")

            all_correct_ind, acc = eval_func(str(updated_md), max_answers)
            accuracies.append(acc)
            all_correct_indices.append(all_correct_ind)
        solution_indices[str(trove).split("/")[-1]]=(all_correct_indices)
    return solution_indices


def load_levels(data_dir):
    """
    Load level info from each JSONL file in the data directory.
    Returns:
        - level_data: category -> list of level numbers (by task index)
        - level_totals: category -> Counter(level -> total count)
    """
    level_data = {}
    level_totals = {}

    data_dir = Path(data_dir)
    for jsonl_file in data_dir.glob("*.jsonl"):
        levels = []
        level_counter = Counter()
        with open(jsonl_file, "r") as f:
            for line in f:
                item = json.loads(line)
                level_str = item.get("level", "")
                if isinstance(level_str, str) and level_str.lower().startswith("level"):
                    try:
                        level_num = int(level_str.split()[-1])
                    except ValueError:
                        level_num = None
                else:
                    level_num = None
                levels.append(level_num)
                if level_num is not None:
                    level_counter[level_num] += 1
        category = jsonl_file.stem
        level_data[category] = levels
        level_totals[category] = level_counter
    return level_data, level_totals


def compute_level_distributions(solution_indices, level_data, level_totals):
    """
    Given solution indices and level metadata, return per-category and per-seed
    level percentages (solved/total).
    """
    level_distributions = defaultdict(list)  # category -> list of dicts (one per seed)

    for category, seeds in solution_indices.items():
        if category not in level_data:
            print(f"Warning: No level data for category {category}")
            continue

        category_levels = level_data[category]
        category_level_totals = level_totals[category]

        for seed_correct_indices in seeds:
            level_counter = Counter()
            for idx in seed_correct_indices:
                if idx is None or idx >= len(category_levels):
                    raise ValueError(f"Index {idx} out of bounds for category {category}")
                    continue
                level = category_levels[idx]
                if level is not None:
                    level_counter[level] += 1

            # Convert counts to percentages
            level_percentage = {
                level: (count / category_level_totals[level]) * 100
                for level, count in level_counter.items()
                if category_level_totals[level] > 0
            }

            level_distributions[category].append(level_percentage)

    return level_distributions

import pandas as pd

def compute_mean_percent_per_level(level_distributions):
    """
    Compute the mean percentage of solved challenges per level for each category.
    Returns a dict: category -> {level -> mean percentage}
    """
    data = {}
    for category, seed_dicts in level_distributions.items():
        all_levels = set()
        for d in seed_dicts:
            all_levels.update(d.keys())

        means = {}
        for level in all_levels:
            level_percentages = [d.get(level, 0.0) for d in seed_dicts]
            means[level] = sum(level_percentages) / len(seed_dicts)

        data[category] = means
    return data