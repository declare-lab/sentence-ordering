# STaCK: Sentence Ordering with Temporal Commonsense Knowledge

This repository contains the pytorch implementation of the paper [STaCK: Sentence Ordering with Temporal Commonsense Knowledge](https://arxiv.org/abs/2109.02247) appearing at EMNLP 2021. 
<!-- The preprint version can be found [here](). -->

![Alt text](stack.png?raw=true "Illustration of STaCK.")

Sentence ordering is the task of finding the correct order of sentences in a randomly ordered document. Correctly ordering the sentences requires an understanding of coherence with respect to the chronological sequence of events described in the text. Document-level contextual understanding and commonsense knowledge centered around these events is often essential in uncovering this coherence and predicting the exact chronological order.  In this paper, we introduce STaCK --- a framework based on graph neural networks and temporal commonsense knowledge to model global information and predict the relative order of sentences. Our graph network accumulates temporal evidence using knowledge of past and future and formulates sentence ordering as a constrained edge classification problem. We report results on five different datasets, and empirically show that the proposed method is naturally suitable for order prediction.

## Data

Contact the authors of the paper [Sentence Ordering and Coherence Modeling using Recurrent Neural Networks](https://arxiv.org/pdf/1611.02654.pdf) to obtain the AAN, NIPS and NSF datasets.

Download the stories of images in sequence SIND dataset (SIS) from the [Visual Storytelling](http://visionandlanguage.net/VIST/dataset.html) website.

Keep the files in appropriate directories in `data/`

The ROC dataset with train, validation, and test splits are provided in this repository.

## Prepare Datasets

Download the COMET model by following instaructions specified in `comet/` directory. Then, run the following:

```
python prepare_data.py
CUDA_VISIBLE_DEVICES=0 python prepare_csk.py
```

## Experiments:

Train and evaluate using:

```
CUDA_VISIBLE_DEVICES=0 python train_csk.py --lr 1e-6 --dataset nips --epochs 10 --hdim 200 --batch-size 8 --pfd
```

For other datasets, you can use the argument `--dataset [aan|nsf|roc|sind]`. The `--pfd` argument ensures that the past and future commonsense knowledge nodes have different relations. Remove this argument to use the same relation. 

We recommend using a learning rate of 1e-6 for all the datasets. Run the experiments multiple times and average the scores to reproduce the results reported in the paper.

## Citation

Please cite the following paper if the use this code in your work:

Deepanway Ghosal, Navonil Majumder, Rada Mihalcea, Soujanya Poria. "STaCK: Sentence Ordering with Temporal Commonsense Knowledge." In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).

## Credits
Some of the code in this repository is borrowed from https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering and https://github.com/allenai/comet-atomic-2020

