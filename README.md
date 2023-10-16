# MagicCoT

This repository contains the code, dataset and models in our paper: MagicLM: Empowering Large Language Models with Improved Chain-of-Thought Fine-Tuning. We release:

- The [44K CoT data](#data-release) generated based on our proposed MagicCoT framework.
- The code for [generating the data](#data-generation-process).
- The code for [fine-tuning](#fine-tune).
- The code for [evaluating the model](#evaluation).
- The code for [CoT debating](#cot-debating).

## Overview

MagicCoT is a Chain-of-Thought improvement framework to synthesize more complicated, diverse, and detailed CoT rationales. In this framework, we introduce three evolution strategies for improving CoT, i.e., complicate, diversify, and specify. 
Following MagicCoT, we generate a large-scale CoT dataset which cotains 44335 samples covering commonsense reasoning, mathematical reasoning, scientific reasoning, and symbolic reasoning. 
Furthermore, we fine-tune open-source LLMs (i.e.,LlaMa 2-Chat 7B and 13B) with our evolved CoT data, called MagicLM, and compare MagicLM to existing popular LLMs on 9 complex reasoning tasks. 
Finally, based on our MagicLM model, we propose a CoT reasoning strategy,step-wise debating.

<p align="center">
  <img src="asset/magiccot.png" alt="MagicCoT framework" width="90%" height="90%">
  <br>
  The Overall Framework of MagicCoT
</p>

## Data Release

The directory [`data`](./data) contains 44k CoT samples generated after 4 rounds based on MagicCoT.
- [`train_data.json`](./data/train_data.json) is all the improved CoT data in the 4 rounds.
- [`no_cs.json`](./data/no_cs.json) is the data after removing commonsense reasoning categories
- [`no_math.json`](./data/no_math.json) is the data after removing mathematicial reasoning categories
- [`no_sci.json`](./data/no_sci.json) is the data after removing scientific reasoning categories
- [`no_sym.json`](./data/no_cs.json) is the data after removing symbolic reasoning categories
- [`seed.json`](./data/seed.json) is the seed dataset used for generation.

## Data Generation Process

Our data generation process is a combination of three pipelines.

- Complicate: Firstly, we use comlicate strategy to complicate the questions of the origin data. Secondly, conduct effect filtering based on the complexity of the new questions. Then, generate answers to new questions. Finally, conduct correctness filtering for new <question, CoT> samples.
- Diversify: Similar to Complicate, but use diversify methods to guide queston generation.
- Specify: First rewrite the CoTs in the seed dataset and then conduct effect filtering.

To perform the generation process using MagicCoT, three scripts [`complicate.sh`, `diversify.sh`, `specify.sh`] are provided in generate.

```
cd generate
bash complicate.sh
bash diversify.sh
bash specify.sh
```

## Fine-tune
We fine-tune LlaMa 2-Chat 7B and 13B models with our dataset. We call the CoT fine-tuning model as MagicLM. The fine-tuning code is adopted from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

```
cd fine-tune
bash run.sh
```

## Evaluation

We conduct evaluation on 9 dataset independent of the seed dataset and present the performance.

```
cd evaluate
bash test.sh
```
<p align="center">
  <img src="asset/evaluation.png" alt="main experimant" width="90%" height="90%">
  <br>
</p>

## CoT Debating

Based on the MagicLM, we propose step-wise CoT debating strategy. To evaluate with CoT debating:

```
cd debate
bash run.sh
```


