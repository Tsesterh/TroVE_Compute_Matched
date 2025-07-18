"""
The file reads the execution-augmented markdown file produced earlier, chooses the "best"
solution for each Example block by using the agreement-based selection algorithm from the original implementation.
It further runs the oracle selection mechanism assuming a perfect selector.
"""

from __future__ import annotations
import json, os, re, sys, textwrap, ast
from pathlib import Path
from typing import Any, List, Dict

EXAMPLE_RE = re.compile(r'##\s*Example\s+(\d+).*?(?=##\s*Example|\Z)', re.S)

EXEC_RE = re.compile(
    r'\*\*(\d+)-th Execution Result\*\*.*?Is Execution Success:\s*(True|False).*?'
    r'Model Answer:\s*\[([^\]]*)\].*?'
    r'(?:-\s*)?Annotated Answer\(s\):\s*\[?([^\]\n\r]*)\]?.*?' # 4  annotated answers (opt)
    r'Is Answer Correct:\s*(True|False)',
    re.S
)


SOL_RE = re.compile(
    r'(\d+)-th \*\*Solution\*\*\n(.*?)\n\*\*\1-th Tools\*\*',
    re.S
)

NUM_RE = re.compile(r'-?\d+\.?\d*')

def unwrap_code(code: str) -> str:
    code = code.replace("```python", "").replace("```", "")
    return textwrap.dedent(code).strip()

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

def select_best_solution(response_list: List[Dict[str, Any]], is_test: bool = True) -> int:
    def get_true_responses(response_list: List[Dict[str, Any]], key: str) -> List[int]:
        return [i for i, r in enumerate(response_list) if r[key]]

    if not is_test:
        correct_indices = get_true_responses(response_list, key="is_correct")
        if len(correct_indices) == 0:
            correct_indices = get_true_responses(response_list, key="is_success")
    else:
        correct_indices = get_true_responses(response_list, key="is_success")
    if len(correct_indices) == 0:
        return 0

    model_answers_dict: Dict[str, List[int]] = {}
    for sidx in correct_indices:
        dres = response_list[sidx]
        for ans in dres["model_answers"]:
            model_answers_dict.setdefault(ans, []).append(sidx)
    if len(model_answers_dict) == 0:
        return correct_indices[0]
    majority_answer, majority_count = "", 0
    for answer, indices in model_answers_dict.items():
        if len(indices) > majority_count:
            majority_answer, majority_count = answer, len(indices)

    majority_response_list = [
        (sidx, response_list[sidx])
        for sidx in model_answers_dict[majority_answer]
    ]

    length_list = []
    for _, dres in majority_response_list:
        try:
            dres_length = get_ast_depth(unwrap_code(dres["solution"]))
        except Exception:
            dres_length = 100
        length_list.append(dres_length)
    index = length_list.index(min(length_list))
    return majority_response_list[index][0]

def parse_example(block: str) -> List[Dict[str, Any]]:
    exec_matches = {m.group(1): m for m in EXEC_RE.finditer(block)}
    responses = []
    for sid, code in SOL_RE.findall(block):
        exec_m = exec_matches.get(sid)
        if not exec_m:
            continue
        _, success, answers_raw, annotated_raw, correct = exec_m.groups()
        answers_list = NUM_RE.findall(answers_raw)
        responses.append({
            "is_success": success == "True",
            "is_correct": correct == "True",
            "model_answers": answers_list,
            "annotated_answers": NUM_RE.findall(annotated_raw or ""),
            "solution":    unwrap_code(code),
        })
    return responses

def build_json(md_text: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    data = []
    oracle_data = []
    examples_with_correct = 0
    total_examples = 0

    for block_m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(block_m.group(1))
        block = block_m.group()
        response_list = parse_example(block)
        if not response_list:
            continue

        total_examples += 1
        correct = any(r["is_correct"] for r in response_list)
        if correct:
            examples_with_correct += 1
        oracle_data.append({
            "example_id": ex_id,
            "is_correct": correct,
        })

        best_idx = select_best_solution(response_list)
        best_res = response_list[best_idx]

        data.append({
            "example_id": ex_id,
            "selected_solution_index": best_idx,
            "is_success": best_res["is_success"],
            "is_correct": best_res["is_correct"],
            "model_answers": best_res["model_answers"],
            "annotated_answers": best_res["annotated_answers"],
            "solution": best_res["solution"],
        })

    oracle_accuracy = examples_with_correct / total_examples if total_examples else 0.0
    return data, oracle_data, oracle_accuracy

def main(in_md: str, out_json: str | None = None) -> None:
    in_path  = Path(in_md)
    out_path = Path(out_json) if out_json else in_path.with_suffix(".selected.json")

    doc = in_path.read_text(encoding="utf-8")

    payload, oracle_data, oracle_accuracy = build_json(doc)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✔  JSON written → {out_path}")

    oracle_out = out_path.with_suffix(".oracle.json")
    oracle_out.write_text(json.dumps(oracle_data, indent=2), encoding="utf-8")
    print(f"✔  Oracle JSON written → {oracle_out}")

    correct_count = sum(1 for item in payload if item["is_correct"])
    total_count = len(payload)
    accuracy = correct_count / total_count if total_count > 0 else 0

    with open(in_md, 'a', encoding='utf-8') as f:
        f.write(f"\n\n### Overall Accuracy: {accuracy:.3f}")
        f.write(f"\n### Oracle Accuracy: {oracle_accuracy:.3f}")
    print(f"✔  Accuracies (regular: {accuracy:.3f}, oracle: {oracle_accuracy:.3f}) appended to markdown file")

if __name__ == "__main__":
    if not (1 < len(sys.argv) < 4):
        sys.exit("Usage: python select_best_from_md.py executions.md [answers.json]")
    main(*sys.argv[1:])