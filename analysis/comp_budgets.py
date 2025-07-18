import re, ast, textwrap
from typing import Any, List, Dict
import numpy as np
from collections import defaultdict
from pathlib import Path


### Regular Expressions to retrieve the candidates from the log data of the .md files
EXAMPLE_RE = re.compile(r'##\s*Example\s+(\d+).*?(?=##\s*Example|\Z)', re.S)

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


def parse_example(block: str) -> List[Dict[str, Any]]:
    """
    Parses a task solution candidate that was extracted from a loggging .md file.
    """
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

def select_best_solution(response_list: List[Dict[str, Any]], is_test: bool = True) -> int:
    """
    Selects the best solution according to the original selection mechanism. This is copied from the original code.
    """
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


def evaluate_markdown_primitive(filename, max_answers = None) -> tuple[List[int], float]:
    """
    Runs the evaluation for a logfile of the primitive baseline with the agreement-based selection mechanism a given computational budget (max_answers).
    """

    with open(filename, "r") as f:
        md_text = f.read()
    
    correct_indices = []
    total = 0

    for block_m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(block_m.group(1))
        block = block_m.group()

        response_list = parse_example(block)
        if not response_list:
            continue

        if len(response_list) != 15:
            breakpoint()

        if max_answers is not None:
            response_list = response_list[:max_answers]

        total += 1
        best_idx = select_best_solution(response_list)
        best_res = response_list[best_idx]

        if best_res["is_correct"]:
            correct_indices.append(ex_id)

    accuracy = len(correct_indices) / total if total else 0.0
    return correct_indices, accuracy

def evaluate_markdown_primitive_oracle(filename , max_answers = None) -> tuple[List[int], float]:
    """
    Runs the evaluation for a logfile of the primitive baseline with the oracle selection mechanism a given computational budget (max_answers).
    """

    with open(filename, "r") as f:
        md_text = f.read()
    
    correct_indices = []
    total = 0

    for block_m in EXAMPLE_RE.finditer(md_text):
        total += 1
        ex_id = int(block_m.group(1))
        block = block_m.group()

        response_list = parse_example(block)
        if not response_list:
            raise ValueError(f"Example {ex_id} has no responses")
        
        if len(response_list) != 5:
            breakpoint()

        if max_answers is not None:
            response_list = response_list[:max_answers]

        # oracle function
        if any(r["is_correct"] for r in response_list):
            correct_indices.append(ex_id)

    accuracy = len(correct_indices) / total if total else 0.0
    return correct_indices, accuracy


def evaluate_markdown_trove(filename, max_answers = None) -> tuple[List[int], float]:
    """
    Runs the evaluation for a logfile of TroVE with the improved agreement-based selection mechanism a given computational budget (max_answers).
    """
    with open(filename, "r") as f:
        md_text = f.read()
    
    modes = ["import", "create", "skip"]
    
    example_blocks: Dict[int, List[str]] = {}
    for m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(m.group(1))
        example_blocks.setdefault(ex_id, []).append(m.group())


    correct_indices = []
    total = 0

    for ex_id, block_list in example_blocks.items():
        total += 1
        # Parse each duplicate block → list[dict]
        per_mode_responses: List[List[Dict[str, Any]]] = [
            parse_example(b) for b in block_list
        ]


        if len(per_mode_responses) != 3:
            breakpoint()
        
        for resp_list in per_mode_responses:
            if len(resp_list) != 5:
                breakpoint()

        # STAGE-1: from each mode, take the first max_answers responses
        for i, resp_list in enumerate(per_mode_responses):
            per_mode_responses[i] = resp_list[:max_answers]

        # Some modes may have 0 parsed responses – filter them paired with mode names
        mode_pairs = [
            (mode, resp) for mode, resp in zip(modes, per_mode_responses) if resp
        ]
        if not mode_pairs:
            # fallback – nothing parsed
            continue

        candidate_list = []
        for mode_name, resp_list in mode_pairs:
            for resp in resp_list:
                candidate_list.append(resp)
            #best_idx = select_best_solution(resp_list)
            #best_resp = resp_list[best_idx].copy()
            #best_resp["mode"] = mode_name
            #candidate_list.append(best_resp)

        # STAGE-2: Now choose among the mode winners
        overall_idx = select_best_solution(candidate_list)
        best_resp = candidate_list[overall_idx]

        if best_resp["is_correct"]:
            correct_indices.append(ex_id)

    accuracy = len(correct_indices) / total if total else 0.0
    return correct_indices, accuracy

def evaluate_markdown_trove_oracle(filename, max_answers = None) -> tuple[List[int], float]:
    """
    Runs the evaluation for a logfile of TroVE with the oracle selection mechanism and a given computational budget (max_answers).
    """
    with open(filename, "r") as f:
        md_text = f.read()
    
    modes = ["import", "create", "skip"]
    
    example_blocks: Dict[int, List[str]] = {}
    for m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(m.group(1))
        example_blocks.setdefault(ex_id, []).append(m.group())


    correct_indices = []
    total = 0

    correct_oracle = 0
    for ex_id, block_list in example_blocks.items():
        total += 1
        # Parse each duplicate block → list[dict]
        per_mode_responses: List[List[Dict[str, Any]]] = [
            parse_example(b) for b in block_list
        ]

        if len(per_mode_responses) != 3:
            breakpoint()
        
        for resp_list in per_mode_responses:
            if len(resp_list) != 5:
                breakpoint()
        
        #from each mode, take the first max_answers responses
        for i, resp_list in enumerate(per_mode_responses):
            per_mode_responses[i] = resp_list[:max_answers]

        # Some modes may have 0 parsed responses – filter them paired with mode names
        mode_pairs = [
            (mode, resp) for mode, resp in zip(modes, per_mode_responses) if resp
        ]
        if not mode_pairs:
            # fallback – nothing parsed
            continue
        
        
        for _, resp_list in mode_pairs:
            if any(r["is_correct"] for r in resp_list):
                correct_oracle += 1
                correct_indices.append(ex_id)
                break
           

    accuracy = len(correct_indices) / total if total else 0.0
    return correct_indices, accuracy


def get_n_accuracies(root_dir, trove_or_prim, max_answers=1):
    """
    Walk `root_dir`, return {trove_dir: {run_name: solved_set}}.
    Only considers *.selected.json* (normal runs).
    """
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]
    
    solutions = defaultdict(dict)

    for trove in trove_dirs:
        all_correct_indices = []
        accuracies = []
        for updated_md in trove.rglob("*.md"):
            if "backup" in updated_md.parts:          # <─ key line
                continue

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
        solutions[str(trove).split("/")[-1]] = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
        }

    return solutions


def run_all_max_answers(root_dir, trove_or_prim, max_n=15):
    """
    Run get_n_accuracies for all budgets from 1 to max_n.
    """
    results = defaultdict(lambda: {"mean": [], "std": []})
    for n in (range(1, max_n + 1)):
        acc = get_n_accuracies(root_dir, trove_or_prim, max_answers=n)
        for category, stats in acc.items():
            results[category]["mean"].append(stats["mean"])
            results[category]["std"].append(stats["std"])
    return results



def get_result_list_primitive(filename):
    """
    Computes the average number of model answers for a primitive baseline run.
    """
    with open(filename, "r") as f:
        md_text = f.read()
    
    correct_indices = []
    total = 0

    model_answer_lenghts = []

    for block_m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(block_m.group(1))
        block = block_m.group()

        response_list = parse_example(block)
        model_answers_set = set()
        for resp in response_list:
            #check if resp["model_answers"] is a empty list
            if not resp["model_answers"]:
                ans = ""
            else:
                if not isinstance(resp["model_answers"], list):
                    ans = resp["model_answers"]
                else:
                    # if it is a list, take the first element
                    ans = resp["model_answers"][0]
                model_answers_set.add(ans)
        model_answer_lenghts.append(len(model_answers_set))

    
    return np.mean(model_answer_lenghts)
            
            

def get_result_list_trove(filename):
    """
    Computes the average number of model answers for a TroVE run.
    """
    with open(filename, "r") as f:
        md_text = f.read()
    
    modes = ["import", "create", "skip"]
    
    example_blocks: Dict[int, List[str]] = {}
    for m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(m.group(1))
        example_blocks.setdefault(ex_id, []).append(m.group())


    correct_indices = []
    total = 0

    model_answer_lengths = []

    for ex_id, block_list in example_blocks.items():
        model_answers_set = set()
        total += 1
        # Parse each duplicate block → list[dict]
        per_mode_responses: List[List[Dict[str, Any]]] = [
            parse_example(b) for b in block_list
        ]


        # Some modes may have 0 parsed responses – filter them paired with mode names
        mode_pairs = [
            (mode, resp) for mode, resp in zip(modes, per_mode_responses) if resp
        ]
        if not mode_pairs:
            # fallback – nothing parsed
            continue
        
        for _, resp_list in mode_pairs:
            for resp in resp_list:
                if not resp["model_answers"]:
                    ans = ""
                else:
                    if not isinstance(resp["model_answers"], list):
                        ans = resp["model_answers"]
                    else:
                        # if it is a list, take the first element
                        ans = resp["model_answers"][0]
                model_answers_set.add(ans)
        model_answer_lengths.append(len(model_answers_set))

    return np.mean(model_answer_lengths)

def collect_primitive_responses_per_task(
    filename: str
) -> Dict[int, List[dict]]:
    """
    Parse `filename` and return every response list for every task.

    Args:
        filename: Markdown file containing the tasks and responses.
        max_answers: If set, truncate each task’s response list to this length.

    Returns:
        Dict[int, List[dict]]: {example_id: response_list}
    """
    with open(filename, "r", encoding="utf-8") as f:
        md_text = f.read()

    responses_per_task: Dict[int, List[dict]] = {}

    for block_m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(block_m.group(1))
        block = block_m.group()

        response_list = parse_example(block)
        if not response_list:
            # Skip examples that have no responses parsed
            continue


        responses_per_task[ex_id] = response_list

    return responses_per_task

def collect_trove_responses_per_task(filename) -> tuple[Dict[int, List[dict]], Dict[int, List[dict]], Dict[int, List[dict]]]:
    """
    Collects the responses for all tasks for all TroVE runs.
    """
    with open(filename, "r") as f:
        md_text = f.read()
    

    responses_import_per_task: Dict[int, List[dict]] = {}
    responses_create_per_task: Dict[int, List[dict]] = {}
    responses_skip_per_task: Dict[int, List[dict]] = {}
    
    example_blocks: Dict[int, List[str]] = {}
    for m in EXAMPLE_RE.finditer(md_text):
        ex_id = int(m.group(1))
        example_blocks.setdefault(ex_id, []).append(m.group())


    for ex_id, block_list in example_blocks.items():
        # Parse each duplicate block → list[dict]
        per_mode_responses: List[List[Dict[str, Any]]] = [
            parse_example(b) for b in block_list
        ]
        
        responses_import_per_task[ex_id] = per_mode_responses[0]
        responses_create_per_task[ex_id] = per_mode_responses[1]
        responses_skip_per_task[ex_id] = per_mode_responses[2]
        
    return responses_import_per_task, responses_create_per_task, responses_skip_per_task

def get_collective_primitive_responses(root_dir):
    """
    Collects the responses for all tasks for all TroVE runs.
    """
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]

    solutions = defaultdict(list)

    for trove in trove_dirs:

        solutions[str(trove).split("/")[-1]] = defaultdict(list)
        for i, updated_md in enumerate(trove.rglob("*.md")):
            if "backup" in updated_md.parts:          # <─ key line
                continue

            eval_func = collect_primitive_responses_per_task

            resp_per_task = eval_func(str(updated_md))
            for j in resp_per_task:
                solutions[str(trove).split("/")[-1]][j].extend(resp_per_task[j])

    return solutions

def get_collective_trove_responses(root_dir):
    """
    Collects the responses for all tasks for all TroVE runs.
    """
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob("*") if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]

    solutions_import = defaultdict(list)
    solutions_create = defaultdict(list)
    solutions_skip = defaultdict(list)

    for trove in trove_dirs:
        solutions_import[str(trove).split("/")[-1]] = defaultdict(list)
        solutions_create[str(trove).split("/")[-1]] = defaultdict(list)
        solutions_skip[str(trove).split("/")[-1]] = defaultdict(list)

        for i, updated_md in enumerate(trove.rglob("*.md")):
            if "backup" in updated_md.parts:          # <─ key line
                continue

            eval_func = collect_trove_responses_per_task

            resp_import, resp_create, resp_skip = eval_func(str(updated_md))
            for j in resp_import:
                solutions_import[str(trove).split("/")[-1]][j].extend(resp_import[j])
            for j in resp_create:
                solutions_create[str(trove).split("/")[-1]][j].extend(resp_create[j])
            for j in resp_skip:
                solutions_skip[str(trove).split("/")[-1]][j].extend(resp_skip[j])
    return solutions_import, solutions_create, solutions_skip

