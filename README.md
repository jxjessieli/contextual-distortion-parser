# Mask Language Models are Implicit Constituency Parsers

This repository contains code for the paper "Contextual Distortion Reveals Constituency: Masked Language Models are Implicit Parsers" in ACL 2023.

## Download Dataset

Since the datasets are not publicly available (e.g. [PTB](https://catalog.ldc.upenn.edu/LDC99T42) and [SPMRL](https://dokufarm.phil.hhu.de/spmrl2013/doku.php?id=start) requires license), we cannot include them in the repo. We provide samples of the data format in the data folder. 


## Quick Start

You can follow the command to parse sentences. Note that it is a straight-forward implementation which processes sentences one-by-one. In the next section, we provide scripts that process batch-by-batch to make the process faster. 
```
python perturb_parse.py --model_name bert-base-uncased --treebank_path ../data/ptb/ptb-dev.txt --pred_tree_path pred_tree
```

## Constituency Parsing
Simply replace the `model_name` argument with the model in Huggingface. For example, if we are using `roberta-base`, we just need to change the `model_name` to `roberta-base`. `layer` specifies the specific layer of pre-trained model from which the representation is used. Use `bert-base-multilingual-uncased` for SPMRL dataset. 
#### Obtain distortion scores with three perturbations
1. Use the following script to obtain distortion scores from the substitution perturbation.

```
python compute_sub.py --model_name bert-base-uncased --treebank_path ../data/ptb/ptb-dev.txt --layer 12 --score_path bert-base-ptb-dev
```

2. Use the following script to obtain distortion scores from the decontextualization perturbation. 

```
python compute_dc.py --model_name bert-base-uncased --treebank_path ../data/ptb/ptb-dev.txt --layer 12 --score_path bert-base-ptb-dev
```

3. Use the following script to obtain distortion scores from the movement perturbation. 

```
python compute_move.py --model_name bert-base-uncased --treebank_path ../data/ptb/ptb-dev.txt --layer 12 --score_path bert-base-ptb-dev
```

#### Normalize scores
Run the following script to normalize distortion scores.
```
python normalize_scores.py --score_path bert-base-ptb-dev --layer 12
```

#### Parse and evaluate
Run the following scripts to do decoding and evaluation using the normalized score.
```
python parse_eval.py --score_path ./normalized_scores/bert-base-ptb_10 --pred_tree_path pred_tree --treebank_path ../data/ptb/ptb-dev-sample.txt
```

## Citation
If you found this work usefule, please cite

```bibtex
@inproceedings{li2023contextual,
      title={Contextual Distortion Reveals Constituency: Masked Language Models are Implicit Parsers}, 
      author={Li, Jiaxi and Lu, Wei},
      booktitle={Proceedings of ACL},
      year={2023}
}
```
