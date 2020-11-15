
# Is Multihop QA in DiRe Condition? <br/> Measuring and Reducing Disconnected Reasoning

This repository contains code for our EMNLP'20 paper "[Is Multihop QA in DiRe Condition? Measuring and Reducing Disconnected Reasoning](https://www.aclweb.org/anthology/2020.emnlp-main.712.pdf)". You can find the corresponding video presentation [here](https://slideslive.com/38939255/is-multihop-reasoning-in-dire-condition-measuring-and-reducing-disconnected-reasoning) and slides [here](http://harshtrivedi.me/resources/dire_emnlp20_final_slides.pdf).

In this work, we propose:

* **DiRe (DIsconnected REasoning) probe**: A way to measure how much disconnected reasoning models do on support annotated multihop reasoning datasets.
* **CSS (Contrastive Support Sufficiency) Transformation**: An automatic transformation of support annotated multifact reasoning datasets to reduce its DiRe cheatability.

## What does it include?

In this code, you can find scripts to i. generate DiRe Probe, CSS Transformation and DiRe Probe of the CSS Transformed HotpotQA, ii. corresponding evaluation scripts and iii. (sample) prediction files from XLNet model described in the paper. If you want to measure DiRe cheatability of your model on HotpotQA or use CSS transformation in your multihop dataset or HotpotQA, this is all you'll need. We'll release our allennlp+huggingface models soon.


## Setup

```bash

# Clone and cd into the repository
git clone https://github.com/stonybrooknlp/dire && cd dire

# Download original hotpotqa
bash download_scripts/download_hotpotqa.sh

# Setup conda env and install dependencies
conda create -n dire python=3.7
conda activate dire
pip install -r basic_requirements.txt
```


## Dataset Conversion

To generate 1. DiRe probe of HotpotQA, 2. CSS transformed HotpotQA and 3. DiRe probe of HotpotQA, run the following script:

```bash
# This will take raw HotpotQA datasets in data/raw and create corresponding
# datasets in (almost) the same format in data/processed/.
python convert_datasets.py
```

Although we have maintained the random seed, it's possible it may behave differently in your machine. If you instead want to download our preprocessed datasets, run the following script and the datasets will be downloaded and stored in `data/processed/`.

```bash
bash download_scripts/download_converted_datasets.sh
```


## Dataset Evaluation

We have 4 evaluation scripts corresponding to 4 types of datasets:

* evaluate_original_dataset.py
* evaluate_probe_of_original_dataset.py
* evaluate_transformed_dataset.py
* evaluate_probe_of_transformed_dataset.py

You can find the sample predictions for these datasets in `sample_predictions/` directory. These predictions are from XLNet model/s described in the paper.

You can run the evaluation scripts of your own model/dataset in the following manner:

```bash
# Evaluate predictions on the original dataset
python evaluation_scripts/evaluate_original_dataset.py \
    data/processed/original_hotpotqa_dev.json \
    sample_predictions/original_hotpotqa_dev_predictions.jsonl

# Evaluate predictions on the DiRe Probe of the original dataset
python evaluation_scripts/evaluate_probe_of_original_dataset.py \
    data/processed/probe_of_original_hotpotqa_dev.json \
    sample_predictions/probe_of_original_hotpotqa_dev_predictions.jsonl \
    sample_predictions/original_hotpotqa_dev_predictions.jsonl

# Evaluate predictions on the CSS Transformed dataset
python evaluation_scripts/evaluate_transformed_dataset.py \
    data/processed/transformed_hotpotqa_dev.json \
    sample_predictions/transformed_hotpotqa_dev_predictions.jsonl

# Evaluate predictions on the DiRe Probe of the CSS Transformed dataset
python evaluation_scripts/evaluate_probe_of_transformed_dataset.py \
    data/processed/probe_of_transformed_hotpotqa_dev.json \
    sample_predictions/probe_of_transformed_hotpotqa_dev_predictions.jsonl
```



## Citing

If you use this work, please cite our paper:

```
@inproceedings{trivedi-etal-2020-multihop,
    title = "Is Multihop {QA} in {DiRe} Condition? Measuring and Reducing Disconnected Reasoning",
    author = "Trivedi, Harsh  and
      Balasubramanian, Niranjan  and
      Khot, Tushar  and
      Sabharwal, Ashish",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.712",
    pages = "8846--8863"
}
```
