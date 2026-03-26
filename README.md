# A Compute-Matched Re-Evaluation of TroVE on MATH

This repository contains the code, analysis, and collected outputs for the paper
["A Compute-Matched Re-Evaluation of TroVE on MATH"](https://arxiv.org/abs/2507.22069).
The work was presented at The second AI for MATH Workshop at the 42nd International
Conference on Machine Learning.
It builds on the original TroVE implementation, which is available
[here](https://github.com/zorazrw/trove).

## Repository Contents

- `data/`: MATH subsets used in this re-evaluation
- `troves/`: collected TroVE runs and evaluation outputs
- `primitives/`: collected PRIMITIVE runs and evaluation outputs
- `evaluate_primitive.py`: evaluation for PRIMITIVE logs
- `evaluate_trove.py`: evaluation for TroVE logs
- `evaluate_trove_corrected.py`: evaluation with the corrected TroVE selection mechanism
- `analysis/`: notebooks and helper scripts for the analysis

## Setup

Set up the environment as in the original TroVE repository. Then place a checkout of
that codebase inside this repository, for example as `trove_main/`, so you can run
the original experiment scripts alongside the evaluation code here.

## Re-Evaluation

To reproduce the re-evaluation, run the original TroVE and baseline experiments for
different seeds. If you only want to inspect the reported outputs, the stored results
are already included in `troves/` and `primitives/`.

### Reproducing TroVE and PRIMITIVE

For TroVE:

```bash
SEED=42
TASK_NAME="math/precalculus"
MODEL_NAME="codellama/CodeLlama-7b-Instruct-hf"

python run_trove.py \
    --seed $SEED \
    --task_name $TASK_NAME \
    --model_name $MODEL_NAME \
    --num_return_sequences 5
```

For PRIMITIVE:

```bash
TASK_NAME="math/precalculus"
SEED=50
SUFFIX="primitive"
NUM_RETURN_SEQUENCES=15
MODEL_NAME="codellama/CodeLlama-7b-Instruct-hf"

python baseline.py \
    --seed $SEED \
    --task_name $TASK_NAME \
    --suffix $SUFFIX \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --model_name $MODEL_NAME
```

To match the number of return sequences, PRIMITIVE needs 3x the number used for TroVE.

### Oracle Evaluation

PRIMITIVE and TroVE can also be evaluated assuming a perfect oracle selection
mechanism. For each `.md` log file from a run, use:

```bash
python evaluate_primitive.py path/to/Primitive_n.md
```

```bash
python evaluate_trove.py path/to/Trove_n.md
```

### Corrected TroVE Selection

To re-evaluate TroVE with the corrected selection mechanism, changing the original
two-stage agreement-based selection into a one-stage selection, run:

```bash
python evaluate_trove_corrected.py path/to/Trove_n.md
```

## Analysis

Run the notebook in `analysis/` to reproduce the reported analysis and figures.
Before doing so, update the `primitive_path` and `trove_path` variables to point to
the PRIMITIVE and TroVE result folders you want to analyze.

## Citation

If you use this repository, please cite:

```bibtex
@article{sesterhenn2025compute,
  title={A Compute-Matched Re-Evaluation of TroVE on MATH},
  author={Sesterhenn, Tobias and Berlot-Attwell, Ian and Zenkner, Janis and Bartelt, Christian},
  journal={arXiv preprint arXiv:2507.22069},
  note={Presented at The second AI for MATH Workshop at the 42nd International Conference on Machine Learning},
  year={2025}
}
```
