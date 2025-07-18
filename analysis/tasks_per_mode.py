from comp_budgets import evaluate_markdown_primitive, evaluate_markdown_trove, evaluate_markdown_trove_oracle, evaluate_markdown_primitive_oracle

from collections import defaultdict
from pathlib import Path
from pathlib import Path
import re


def get_n_solved(root_dir, trove_or_prim, max_answers=1):
    """
    Walk `root_dir`, return {trove_dir: {run_name: solved_set}}.
    Only considers *.updated.selected.json* (normal runs).
    """
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]
    
    solutions = defaultdict(dict)

    for trove in trove_dirs:
        all_correct_indices = []
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

            all_correct_index, _ = eval_func(str(updated_md), max_answers)
            all_correct_indices.append(all_correct_index)
        solutions[str(trove).split("/")[-1]] = all_correct_indices
    return solutions

##### for TROVE ####

EX_BLOCK  = re.compile(r"##\s*Example\s+(\d+)(.*?)(?=##\s*Example|\Z)", re.S | re.I)
CORRECT   = re.compile(r"Is Answer Correct:\s*True", re.I)


def _trove_parse(md_path):
    ORDER, seen = ("import", "create", "skip"), {}
    solved = {m: set() for m in ORDER}
    text   = Path(md_path).read_text(encoding="utf-8")
    for ex_str, blk in EX_BLOCK.findall(text):
        ex   = int(ex_str)
        idx  = seen.get(ex, 0)
        if idx < 3:
            mode = ORDER[idx]
            if CORRECT.search(blk):
                solved[mode].add(ex)
        seen[ex] = idx + 1
    return solved


def get_trove_solved(root_dir):
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]

    solutions = defaultdict(lambda: defaultdict(list))

    for trove in trove_dirs:
        for updated_md in trove.rglob("*.md"):
            if "backup" in updated_md.parts:
                continue
            parsed = _trove_parse(updated_md) #looks like {"import": [1,5,6,9], "create:" [2,3,4], "skip": [7,8]}
            for mode in ("import", "create", "skip"):
                if parsed[mode]:  # only append if non-empty
                    solutions[str(trove).split("/")[-1]][mode].append(sorted(parsed[mode]))
    return solutions