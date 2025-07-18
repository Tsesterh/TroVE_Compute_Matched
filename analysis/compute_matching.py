import json
from pathlib import Path

def read_accuracy(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if not data:
            return 0.0
        correct = sum(1 for d in data if d.get("is_correct", False))
        return correct / len(data)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read or parse {file_path}")
        return None

def get_trove_accuracies(root_dir):
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob('*') if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]

    trove_accuracies = {}

    for trove_dir in trove_dirs:
        updated_files = list(trove_dir.rglob('*.md'))
        for updated_file in updated_files:
            if "backup" in updated_file.parts:
                print("Skipping backup file:", updated_file)
                continue
            base_path = updated_file.with_suffix('')
            normal_file = base_path.with_suffix('.selected.json')
            new_eval_file = base_path.with_suffix('.selected_new_eval.json')
            oracle_file = base_path.with_suffix('.selected.oracle.json')

            normal_acc = read_accuracy(normal_file)
            new_eval_acc = read_accuracy(new_eval_file)
            oracle_acc = read_accuracy(oracle_file)

            if normal_acc is None or new_eval_acc is None or oracle_acc is None:
                continue

            trove_key = str(trove_dir)
            if trove_key not in trove_accuracies:
                trove_accuracies[trove_key] = {"normal": [], "new_eval": [], "oracle": []}

            trove_accuracies[trove_key]["normal"].append(normal_acc)
            trove_accuracies[trove_key]["new_eval"].append(new_eval_acc)
            trove_accuracies[trove_key]["oracle"].append(oracle_acc)

    return trove_accuracies


def get_primitive_accuracies(root_dir):
    root_dir = Path(root_dir)
    trove_dirs = [d for d in root_dir.rglob('*') if d.is_dir()]
    #remove dirs called backup
    trove_dirs = [d for d in trove_dirs if "backup" not in str(d)]

    trove_accuracies = {}
    for trove_dir in trove_dirs:
        updated_files = list(trove_dir.rglob('*.md'))
        for updated_file in updated_files:
            if "backup" in updated_file.parts:
                print("Skipping backup file:", updated_file)
                continue
            base_path = updated_file.with_suffix('')
            normal_file = base_path.with_suffix('.selected.json')
            oracle_file = base_path.with_suffix('.selected.oracle.json')

            normal_acc = read_accuracy(normal_file)
            oracle_acc = read_accuracy(oracle_file)

            if normal_acc is None or oracle_acc is None:
                continue

            trove_key = str(trove_dir)
            if trove_key not in trove_accuracies:
                trove_accuracies[trove_key] = {"normal": [], "oracle": []}

            trove_accuracies[trove_key]["normal"].append(normal_acc)
            trove_accuracies[trove_key]["oracle"].append(oracle_acc)

    return trove_accuracies