import argparse
from utils.eval_for_comparison import get_stats
from utils.trees_processing import load_ptb
from tqdm import tqdm

import pickle
import numpy as np
from pathlib import Path

def tree_from_scores(span_scores, sentence):
    chart = {}
    for length in range(1, len(sentence) + 1):
        for left in range(0, len(sentence) + 1 - length):
            right = left + length
            if length == 1:
                chart[left, right] = [], span_scores[left][right-1]
                continue
            if length == 2:
                chart[left, right] = [], chart[left, left+1][1] + chart[left+1, right][1] + span_scores[left][right-1]
                continue

            best_split = min(
                range(left + 1, right),
                key=lambda split: (chart[left, split][1] + chart[split, right][1] + span_scores[left][right-1]),
            )

            left_spans, left_score = chart[left, best_split]
            right_spans, right_score = chart[best_split, right]
            children = left_spans + right_spans + [(left, best_split-1)] + [(best_split, right-1)]

            chart[left, right] = (children, span_scores[left][right-1] + left_score + right_score)

    split_ids, score = chart[0, len(sentence)]
    split_ids.append((0, len(sentence)-1))
    return  split_ids

def post_process(const_spans):
    '''
    remove length=1 spans and (0, sent_len-1) span
    '''
    processed = set()
    const_spans = const_spans[:-1]      ## Remove (0, sent_len-1)
    for (left, right) in const_spans:
        if left != right:               ## Remove length=1 spans
            processed.add((left, right))
    return processed

def run(config):
    test_ptb_path = config.treebank_path
    test_data = load_ptb(test_ptb_path, lower=True)
    print("test data loaded")
    
    ## Load distortion scores
    with open(config.score_path, 'rb') as f:
        scores = pickle.load(f)

    ## Decode parse trees and evaluate
    corpus_f1 = [0.0, 0.0, 0.0]
    sent_f1 = []
    pred_all = []

    labels = ['sbar', 'np', 'vp', 'pp', 'adjp', 'advp', 's']
    label_total = {label: 0 for label in labels}
    label_correct = {label: 0 for label in labels}

    for i in tqdm(range(len(test_data))):
        distortion_score = scores[i]
        example = test_data[i]
        sent = test_data[i]['sent']
        pred_spans = tree_from_scores(distortion_score, sent)
        pred_all.append(pred_spans)
        pred_set = post_process(pred_spans)
        
        gold_tags = example['gold_tags']
        gold_spans = example['gold_spans']
        gold_set = set(gold_spans)
        if (0, len(sent)-1) in gold_set:
            gold_set.remove((0, len(sent)-1))

         ## Update corpus F1
        tp, fp, fn = get_stats(pred_set, gold_set)
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn

        ## Update sentence F1
        overlap = pred_set.intersection(gold_set)
        prec = float(len(overlap)) / (len(pred_set) + 1e-8)
        reca = float(len(overlap)) / (len(gold_set) + 1e-8)
        if len(gold_set) == 0:
            reca = 1.
            if len(pred_set) == 0:              
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.append(f1)

        ## Update label recalls
        for i, tag in enumerate(gold_tags):
            if tag not in labels:
                continue
            label_total[tag] += 1
            if gold_spans[i] in pred_spans:
                label_correct[tag] += 1

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision", prec)
    print("Recall: ", recall)
    corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
    print("corpus F1: ", corpus_f1)
    sent_f1_mean = np.mean(np.array(sent_f1))
    print("sentence F1: ", sent_f1_mean)

    for i in range(len(labels)):
        if label_total[labels[i]] == 0:
            continue
        print(labels[i], label_correct[labels[i]]/label_total[labels[i]])

    ## Write trees to file
    Path("./pred_spans/").mkdir(parents=True, exist_ok=True)
    with open('./pred_spans/' + config.pred_tree_path, "w") as outfile:
        # outfile.write("\n".join(pred_all))
        for pred in pred_all:
            outfile.write(f"{pred}\n")

    ## Write performance to file
    Path("./performance/").mkdir(parents=True, exist_ok=True)
    with open("./performance/" + config.pred_tree_path, "w") as outfile:
        outfile.write(f"Precision: {prec}\n")
        outfile.write(f"Recall: {recall}\n")
        outfile.write(f"Corpus F1: {corpus_f1}\n")
        outfile.write(f"Sentence F1: {sent_f1_mean}\n")
        for i in range(len(labels)):
            if label_total[labels[i]] == 0:
                continue
            outfile.write(f"{labels[i]}: {label_correct[labels[i]]/label_total[labels[i]]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--score_path', type=str, default='bert-base-ptb')
    parser.add_argument('--pred_tree_path', type=str, default='pred_tree.txt')
    parser.add_argument('--treebank_path', type=str, default="/../data/spmrl_cleaned/hungarian-test.txt")
    config = parser.parse_args()
    run(config)
    
if __name__ == '__main__':
    main()