"""
This file provides the functionality to read a given markdown file and evaluate it by using the agreement-based selection algorithm from the original implementation.
It further runs the oracle selection mechanism assuming a perfect selector.
"""

from __future__ import annotations
import json, os, re, sys, textwrap, ast, argparse
from pathlib import Path
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────
# 1.  Regex helpers (same as primitive)                         
# ──────────────────────────────────────────────────────────────
EXAMPLE_RE = re.compile(r'##\s*Example\s+(\d+).*?(?=##\s*Example|\Z)', re.S)

# Now also captures an *optional* Annotated Answer(s) line
EXEC_RE = re.compile(
    r'\*\*(\d+)-th Execution Result\*\*.*?'                   # 1  solution id
    r'Is Execution Success:\s*(True|False).*?'                # 2  success flag
    r'Model Answer:\s*\[([^\]]*)\].*?'                        # 3  model answers
    r'(?:-\s*)?Annotated Answer\(s\):\s*\[?([^\]\n\r]*)\]?.*?'# 4  annotated answers (opt)
    r'Is Answer Correct:\s*(True|False)',                     # 5  correctness flag
    re.S,
)

SOL_RE = re.compile(
    r'(\d+)-th \*\*Solution\*\*\n(.*?)\n\*\*\1-th Tools\*\*',
    re.S
)

NUM_RE  = re.compile(r'-?\d+\.?\d*')

# ──────────────────────────────────────────────────────────────
# 2.  Code helpers                                              
# ──────────────────────────────────────────────────────────────

def unwrap_code(code: str) -> str:
    code = code.replace("```python", "").replace("```", "")
    return textwrap.dedent(code).strip()


def _depth(node: ast.AST, lvl: int = 0) -> int:
    return max([lvl] + [_depth(c, lvl + 1) for c in ast.iter_child_nodes(node)])


def get_ast_depth_2(code: str) -> int:
    try:
        tree = ast.parse(code)
        return _depth(tree)
    except SyntaxError:
        return 100  # treat junk as long trajectory
    
def get_ast_depth(code: str) -> int:
    root = ast.parse(code)
    total_depth = 0

    def depth_ast(root): 
        return 1 + max((depth_ast(child) for child in ast.iter_child_nodes(root)), default = 0)
    
    for node in root.body:
        if isinstance(node, ast.FunctionDef):
            continue
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant):
                continue
            elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'df':
                continue
        depth = depth_ast(node)
        total_depth += depth
    return total_depth

# ──────────────────────────────────────────────────────────────
# 3.  Selection algorithm (same as the original implementation, we do not change it)                         
# ──────────────────────────────────────────────────────────────

def select_best_solution(response_list: List[Dict[str, Any]], *, is_test: bool = True) -> int:
    """Select the best solution among multiple predictions."""

    def _indices_where(key: str) -> List[int]:
        return [i for i, r in enumerate(response_list) if r[key]]

    if not is_test:
        correct_idx = _indices_where("is_correct") or _indices_where("is_success")
    else:
        correct_idx = _indices_where("is_success")
    if not correct_idx:
        return 0

    # majority answer among the surviving indices
    answer_groups: Dict[str, List[int]] = {}
    for idx in correct_idx:
        for ans in response_list[idx]["model_answers"]:
            answer_groups.setdefault(ans, []).append(idx)
    if not answer_groups:
        return correct_idx[0]
    majority_answer = max(answer_groups.items(), key=lambda it: len(it[1]))[0]
    majority_indices = answer_groups[majority_answer]

    # choose the one with shortest trajectory
    traj_lengths = []
    for idx in majority_indices:
        try:
            traj_len = get_ast_depth(unwrap_code(response_list[idx]["solution"]))
        except Exception:
            traj_len = 100
        traj_lengths.append(traj_len)
    best_local = majority_indices[traj_lengths.index(min(traj_lengths))]
    return best_local

# ──────────────────────────────────────────────────────────────
# 4.  Markdown parsing                                           
# ──────────────────────────────────────────────────────────────

def parse_example_block(block: str) -> List[Dict[str, Any]]:
    exec_matches = {m.group(1): m for m in EXEC_RE.finditer(block)}
    sol_matches = {sid: code for sid, code in SOL_RE.findall(block)}
    responses = []
    
    for sid, exec_m in exec_matches.items():
        _, success, answers_raw, annotated_raw, correct = exec_m.groups()
        answers_list = NUM_RE.findall(answers_raw)
        
        # Get solution code if available, otherwise use empty string or None
        solution_code = sol_matches.get(sid, "")
        
        responses.append({
            "is_success": success == "True",
            "is_correct": correct == "True",
            "model_answers": answers_list,
            "annotated_answers": NUM_RE.findall(annotated_raw or ""),
            "solution": unwrap_code(solution_code) if solution_code else "",
        })
    return responses

# ──────────────────────────────────────────────────────────────
# 5.  Build final JSON                                           
# ──────────────────────────────────────────────────────────────

def build_json(md_text: str, modes: List[str]) -> tuple[List[Dict[str, Any]], float]:
    # 1) gather all blocks per example id (order preserved → mode index)
    example_blocks: Dict[int, List[str]] = {}
    for m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(m.group(1))
        example_blocks.setdefault(ex_id, []).append(m.group())

    data = []
    oracle_data = []
    total_examples = len(example_blocks)
    examples_with_correct = 0

    for ex_id, block_list in example_blocks.items():
        # Parse each duplicate block → list[dict]
        per_mode_responses: List[List[Dict[str, Any]]] = [
            parse_example_block(b) for b in block_list
        ]

        for i in per_mode_responses:
            if len(i) != 5:
                breakpoint()
            else:
                print("ok")
        # Some modes may have 0 parsed responses – filter them paired with mode names
        mode_pairs = [
            (mode, resp) for mode, resp in zip(modes, per_mode_responses) if resp
        ]
        if not mode_pairs:
            # fallback – nothing parsed
            print(f"No parsed responses for example {ex_id}")
            continue

        

        candidate_list = []
        for mode_name, resp_list in mode_pairs:
            #if ex_id == 94:
            #    breakpoint()
            best_idx = select_best_solution(resp_list)
            best_resp = resp_list[best_idx].copy()
            best_resp["mode"] = mode_name
            candidate_list.append(best_resp)



        # Now choose among the mode winners
        overall_idx = select_best_solution(candidate_list)
        best_resp = candidate_list[overall_idx]
        #breakpoint()
        correct = False
        # accuracy bookkeeping
        for resp_list in per_mode_responses:
            if any(r["is_correct"] for r in resp_list):
                correct = True
                break
        examples_with_correct += 1 if correct else 0


        data.append({
            "example_id":         ex_id,
            "selected_mode":      best_resp.get("mode", ""),
            "is_success":         best_resp["is_success"],
            "is_correct":         best_resp["is_correct"],
            "model_answers":      best_resp["model_answers"],
            "annotated_answers":  best_resp["annotated_answers"],
            "solution":           best_resp["solution"],
        })

        oracle_data.append({
            "example_id":         ex_id,
            "is_correct": correct,
        })

    oracle_accuracy = examples_with_correct / total_examples if total_examples else 0.0
    print("Total examples: ", total_examples)
    print("Examples with correct: ", examples_with_correct)
    print("Oracle accuracy: ", oracle_accuracy)
    return data, oracle_data, oracle_accuracy


def main(md_file: str, out_json: str | None, modes: List[str]):
    in_p  = Path(md_file)
    out_p = Path(out_json) if out_json else in_p.with_suffix(".selected.json")

    md_text = in_p.read_text(encoding="utf-8")

    payload, oracle_data, oracle_acc = build_json(md_text, modes)

    # save JSON
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✔  JSON written → {out_p}")

    # save oracle data
    oracle_out = out_p.with_suffix(".oracle.json")
    oracle_out.write_text(json.dumps(oracle_data, indent=2), encoding="utf-8")
    print(f"✔  Oracle JSON written → {oracle_out}")

    # regular accuracy (best_resp correctness)
    correct = sum(1 for item in payload if item["is_correct"])
    total   = len(payload)
    acc = correct / total if total else 0.0
    print("Non-Oracle")
    print("Total examples: ", total)
    print("Correct examples: ", correct)
    print("Accuracy: ", acc)

    with open(md_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n### Overall Accuracy: {acc:.3f}")
        f.write(f"\n### Oracle Accuracy: {oracle_acc:.3f}")
    print(f"✔  Accuracies (overall: {acc:.3f}, oracle: {oracle_acc:.3f}) appended to markdown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best solutions from trove markdown file and output JSON.")
    parser.add_argument("markdown", help="Input markdown file with execution results")
    parser.add_argument("json_out", nargs="?", help="Optional output JSON path")
    parser.add_argument("--modes", nargs="*", default=["import", "create", "skip"], help="Mode names in the order they appear (default: import create skip)")
    args = parser.parse_args()

    main(args.markdown, args.json_out, args.modes)