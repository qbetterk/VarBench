# VarBench: Robust Language Model Benchmarking Through Dynamic Variable Perturbation
[[Paper]](https://arxiv.org/abs/2406.17681)|[[Huggingface]](https://huggingface.co/datasets/Columbia-NLP/VarBench)
VarBench is a new benchmark with dynamically-valued variables to help deal with the problem of benchmark data contamination. Currently, the following tasks are supported: **[GSM8K](https://arxiv.org/abs/2110.14168)**, **[CommonsenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa)**, **[AI2 Reasoning Challenge](https://allenai.org/data/arc)**, and **[TruthfulQA](https://arxiv.org/abs/2109.07958)**. We are planning to extend VarBench to include other complex tasks such as **[AGIEval](https://arxiv.org/abs/2304.06364)** and **[MMLU](https://arxiv.org/pdf/2009.03300)**.
To keep the results comparable, we use the [lm-eval harness](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI.

1. [Constructing VarBench](#varbench)
	- [GSM+](#gsm8k)
	- [TruthfulQA+](#tqa)
	- [CommonsenseQA+  (CSQA+)](#csqa)
	- [ARC+](#arc)
2. [Extracting Variables](#extract)
3. [Alternative Perturbation](#others)
4. [Citation](#citation)

## Constructing VarBench <a name="varbench"></a>
The default extracted variables are stored in ``./gen_data/${dataset}/${split}_${dataset}_${model}.jsonl``, where ``dataset`` is one of ``gsm8k, truthfulqa, csqa, arc_challenge``, ``split`` is one of ``test, dev, validation`` (depending on each dataset), and ``model`` is on default ``gpt4o`` in this work. In this section we will use them to construct new test sets.
> **Note:** These files has been manually corrected and verified. Therefore, try not to overwrite these files.

#### Constructing GSM+ <a name="gsm8k"></a>

we need to sample variable values. By selecting different random seeds, we can create a new set of values for the variables defined in the previous step.
```sh
python sample.py \
    --data_path ./gen_data/gsm8k/test_gsm8k_gpt4o.jsonl \
    --save_dir ./gen_data/gsm8k/sample_42 \
    --task generate_test_set_gsm8k \
    --seed 42
```

  

#### Constructing TruthfulQA+ <a name="tqa"></a>
```sh
python sample.py \
    --data_path ./gen_data/truthfulqa/validation_truthfulqa_gpt4o.jsonl \
    --save_dir ./gen_data/truthfulqa/sample_42 \
    --task generate_dev_set_csqa \
    --seed ${seed}
```

#### Constructing CommonsenseQA+ (CSQA+) <a name="csqa"></a>

```sh
python sample.py \
    --data_path ./gen_data/csqa/dev_csqa_gpt4o.jsonl \
    --save_dir ./gen_data/csqa/sample_42 \
    --task generate_dev_set_csqa \
    --seed ${seed}
```

#### Constructing ARC+ <a name="arc"></a>
```sh
python sample.py \
    --data_path ./gen_data/arc/challenge/test_arc_challenge_gpt4o.jsonl \
    --save_dir ./gen_data/arc/challenge/sample_42 \
    --task generate_test_set_arc \
    --seed ${seed}
```

## Extracting Variables <a name="extract"></a>

>This step is optional if you only wish to select new variable values for the benchmark

The first step is to extract variables from the original GSM8K test set to create a delexicalized version, and construct code solutions for each problem.
```sh
python generate.py \
    --model_name_or_path "gpt-4o" \
    --top_p 0.3 \
    --save_dir ./gen_data/gsm8k \
    --save_filename gsm8k_test_gpt4o.jsonl \
    --task generate_gsm8k
```
This step will add three components for each datapoint, under keys ``variables, question_delex, func`` correspondingly.
>**Note:** that the save_filename is different from the default ``test_gsm8k_gpt4o.jsonl``, avoiding overwrite the original annotation.

  

Then we prompt gpt again to create value range for each variable. On default, we load variables from ``gsm8k_test_gpt4o.jsonl``, which is generated in the previous step.
```sh
python generate.py \
    --model_name_or_path "gpt-4o" \
    --top_p 0.3 \
    --save_dir ./gen_data/gsm8k \
    --save_filename gsm8k_test_gpt4o_range.jsonl \
    --task generate_input_range
```
This step will add component under key ``input_range`` for each data point.

## Alternative Perturbation <a name="others"></a>
Alternative perturbations are created in ``other_process.py`` file.
- For ``gsm8k``, we conduct paraphrasing:
```sh

python other_process.py \
	--task paraphrase_gsm8k \
	--save_dir ./gen_data/gsm8k/paraphrase/ \
	--save_filename "test.jsonl" \
	--seed 2
```
- For ``arc`` and ``csqa`` we conduct shuffling:
```sh
python other_process.py \
	--task shuffle_arc \
	--save_dir "./gen_data/arc/challenge/shuffle/" \
	--save_filename "test.jsonl" \
	--seed 40
```
But for ``csqa``, we conduct shuffling during sampling by change ``line 144`` in ``sample.py`` to be ``True``.
- For ``truthfulqa``, we rewrite questions by setting ``new_question`` in ``line 293`` in ``sample.py`` to be ``True``.
  

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
