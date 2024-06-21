# VarBench: Robust Language Model Benchmarking Through Dynamic Variable Perturbation
VarBench is a new benchmark with dynamically-valued variables to help deal with the problem of benchmark data contamination. Currently, the following tasks are supported: **GSM-8K, CommonsenseQA, AI2 Reasoning Challenge, and TruthfulQA**. We are planning to extend VarBench to include other complex tasks such as MATH and MMLU.
To keep the results comparable, we use the [lm-eval harness](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI.

1. [Constructing VarBench](#varbench)
    1. [Constructing GSM+](#gsm8k)
    2. [Constructing TruthfulQA+](#tqa)
    3. [Constructing CommonsenseQA+](#cqa)
    4. [Constructing ARC+](#arc)
3. [Citation](#citation)

## Constructing VarBench <a name="varbench"></a>

### Constructing GSM+ <a name="gsm8k"></a>

(This step is optional if you only wish to select new variable values for the benchmark)
The first step is to extract variables from the original GSM-8K test set to create a delexicalized version, and construct code solutions for each problem.
```sh
python generate.py \
        --model_name_or_path "gpt-4-turbo" \
        --top_p 0.3 \
        --save_filename "GSM8K_test.jsonl" 
```

Next, we need to sample variable values. By selecting different random seeds, we can create a new set of values for the variables defined in the previous step.
```
python sample.py \
        --model_name_or_path "gpt-4-turbo" \
        --seed 42
```

### Constructing TruthfulQA+ <a name="tqa"></a>

### Constructing CommonsenseQA+ <a name="cqa"></a>

### Constructing ARC+ <a name="arc"></a>

## Citation <a name="citation"></a>

If you end up using this benchmark or the accompanying code, please cite the following paper.
```
@inproceedings{qian2024varbench,
    title = "{VarBench}: Robust Language Model Benchmarking Through Dynamic Variable Perturbation",
    author = "Qian, Kun  and
      Wan, Shunji and
      Tang, Claudia and
      Wang, Youzhi and
      Zhang, Xuanming and
      Chen, Maximillian  and
      Yu, Zhou",
    booktitle = "arXiv preprint",
    month = june,
    year = "2024",
}
```
