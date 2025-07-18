# A Compute-Matched Re-Evaluation of TroVE on MATH

This work containts the implementation of the AI4MATH workshop paper.
It bases on the original implementation of TroVE, which can be found [here](https://github.com/zorazrw/trove).

## Setup
We recommend to perform the setup according to the original implementation. We recommend to download it and include it in this repository (e.g. as a `trove_main` folder).

## Re-Evaluation
To re-evaluate, run the original TroVE and baseline experiments for different seeds.
Instead, you can also use our results, which are provided in the `troves` and `primitives` folders.

### Reproducing TroVE and Primitive

For TroVE:

```bash
SEED=42
TASK_NAME="math/precalculus"
MODEL_NAME="codellama/CodeLlama-7b-Instruct-hf"

python run_trove.py \
    --seed $SEED \
    --task_name $TASK_NAME \
    --model_name $MODEL_NAME\
    --num_return_sequences 5
```

For Primitive:

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

Note that to match the number of return sequences, Primitive needs 3x the number of TroVE.

### Evaluating with Oracle mode

Primitive and TroVE are also evaluated assuming a perfect oracle selection mechanism. 
For each .md log file creating the logs of a run, run the following:

```python
python evaluate_primitive.py Primitive_n.md
```

```python
python evaluate_trove.py Trove_n.md
````

### Correcting/Improving TroVE's selection mechanism
To correct TroVE's selection mechanism from a two-stage aggrement-based selection to a one-stage selection, run the following:

```python
python evaluate_trove_corrected.py Trove_n.md
```

## Analysis
In the analysis folder, you can then run the analysis notebook to perform the evaluation.

Make sure to correct the `primitive_path` and `trove_path`variables to point to the Primitive and TroVE folders.
