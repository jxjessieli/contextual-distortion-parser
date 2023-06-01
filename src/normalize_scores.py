import pickle
import torch
from sklearn import preprocessing
from pathlib import Path
import os
import argparse

def normalize_score(span_scores):
    normalized_scores = torch.zeros_like(span_scores)
    for i in range(span_scores.shape[0]):
        normalized_vector = torch.tensor(preprocessing.normalize(torch.diag(span_scores, i).unsqueeze(0)))
        normalized_scores += torch.diag_embed(normalized_vector, offset=i).squeeze()
    return normalized_scores

def run(config):
    layer = config.layer
    with open(f'./scores/sub_{config.score_path}_{layer}', 'rb') as f:
        sub = pickle.load(f)
    with open(f'./scores/dc_{config.score_path}_{layer}', 'rb') as f:
        dc = pickle.load(f)
    with open(f'./scores/move_{config.score_path}_{layer}', 'rb') as f:
        move = pickle.load(f)

    ## post-process scores
    processed = []
    for i in range(len(move)):
        processed_i = torch.zeros_like(move[i])
        sent_len = move[i].shape[0]
        for length in range(1, sent_len):
            for left in range(0, sent_len+1-length):
                right = left+length
                if left == 0 and right == sent_len:
                    continue
                elif left == 0 or right == sent_len:
                    processed_i[left][right-1] = (move[i][left][right-1] * 4 + sub[i][left][right-1] + dc[i][left][right-1])/6
                else:
                    processed_i[left][right-1] = (move[i][left][right-1] * 6 + sub[i][left][right-1] + dc[i][left][right-1])/8
        processed.append(processed_i)

    ## normalize scores
    normalized_all = []
    for i in range(len(move)):
        normalized = normalize_score(processed[i])
        normalized_all.append(normalized)

    Path(f'./normalized_scores/').mkdir(parents=True, exist_ok=True)
    with open(f'./normalized_scores/{config.score_path}_{layer}', 'wb') as file:
        pickle.dump(normalized_all, file) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--score_path', type=str, default='bert-base-ptb')

    config = parser.parse_args()
    run(config)

if __name__ == '__main__':
    main()