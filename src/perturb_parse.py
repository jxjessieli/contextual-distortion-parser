import torch
from transformers import RobertaTokenizerFast, RobertaModel, BertTokenizerFast, BertModel

from utils.trees_processing import load_ptb
from utils.eval_for_comparison import get_stats
import numpy as np
from tqdm import tqdm
import argparse
from sklearn import preprocessing
import copy
from pathlib import Path

from utils.token_embed import embed_token_mean

class PerturbParse(torch.nn.Module):
    def __init__(self, model_name, device, layer):
        super().__init__()
        self.model_name = model_name
        if 'roberta' in model_name:
            print('Roberta model: {}'.format(model_name))
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            self.bert = RobertaModel.from_pretrained(model_name)
            self.MASK = '<mask>'
        else:
            print('Bert model: {}'.format(model_name))
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            self.bert = BertModel.from_pretrained(model_name)
            self.MASK = '[MASK]'

        self.layer = layer
        self.to(device)
        self.device = device

    def tree_from_split_scores(self, span_scores, sentence):
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
        return  split_ids

    def normalize_score(self, span_scores):
        normalized_scores = torch.zeros_like(span_scores)
        for i in range(span_scores.shape[0]):
            normalized_vector = torch.tensor(preprocessing.normalize(torch.diag(span_scores, i).unsqueeze(0)))
            normalized_scores += torch.diag_embed(normalized_vector, offset=i).squeeze()
        return normalized_scores

    def forward(self, sentence):
        '''
        find the tree with the maximum expected number of constituents
        sentence: a list of tokens
        '''
        if len(sentence) == 1 or len(sentence) == 2:
            return []

        ## Produce a list of spans
        span_ids = []
        sent_len = len(sentence)
        for i in range(sent_len):
            for j in range(i+1, sent_len+1):
                span_ids.append((i, j))

        ## Obtain scores for each span
        span_probs = self.compute_score(sentence, span_ids)

        # Normalize scores
        normalized_span_probs = self.normalize_score(span_probs)

        ## Perform CKY to get the best tree
        tree = self.tree_from_split_scores(normalized_span_probs, sentence)
        tree.append((0, sent_len-1))
        return tree

    def get_output_reprs(self, sentence, span_id):
        sentence_str = ' '.join(sentence)
        input = self.tokenizer(sentence_str, return_tensors='pt', return_offsets_mapping=True).to(self.device)
        # tokenized = self.tokenizer.convert_ids_to_tokens(input['input_ids'].squeeze().cpu().tolist())
        # output = self.bert(**input).last_hidden_state.squeeze().detach()
        output_all = self.bert(input_ids=input.data['input_ids'], attention_mask=input.data['attention_mask'], output_hidden_states=True)
        output = output_all.hidden_states[self.layer].data.squeeze().detach()
        output_reprs = embed_token_mean(sentence_str, input[0], output)
        if span_id == None:
            return output_reprs
        else:
            ## return span representations  [span_len, hidden_dim]
            left, right = span_id
            return output_reprs[left:right, :]

    def calc_distortion(self, gold_reprs, mask_reprs):
        """
        Args:
            gold_reprs: representations from the original sentence [num_tokens, hidden_dim]
            mask_reprs: representations from the masked sentence [num_tokens, hidden_dim]
        """
        num_tokens = gold_reprs.shape[0]
        distortion = torch.norm(gold_reprs - mask_reprs)**2 / num_tokens
        return distortion

    def compute_score(self, sentence, span_ids):
        """Obtain scores from perturbation for each span

        Args:
            sentence: a list of tokens
            span_ids: a list of span indices, sentence[i:j] is a valid span
        
        Returns:
            span_scores: a chart of scores of shape (sentence length, sentence length, label vocab size)
        """
        sent_len = len(sentence)
        span_scores = torch.zeros(sent_len, sent_len)
        gold_reprs = self.get_output_reprs(sentence, None)  # [seq_len, hidden_dim]
        for (left, right) in span_ids:
            if left == 0 and right == sent_len:
                continue
            elif left == 0:
                inside_masked = sentence[:left] + [self.MASK] + sentence[right:]
                inside_reprs = self.get_output_reprs(inside_masked, None)

                front_move_sent = sentence[left:right] + [','] + sentence[right:]
                end_move_sent = sentence[right:] + [','] + sentence[left:right]
                front_move_indices = [i for i in range(sent_len+1) if i != (right-left)]
                end_move_indices = [i for i in range(sent_len+1) if i != (sent_len-right)]

            elif right == sent_len:
                inside_masked = sentence[:left] + [self.MASK] + sentence[right:]
                inside_reprs = self.get_output_reprs(inside_masked, None)

                front_move_sent = sentence[left:right] + [','] + sentence[:left]
                end_move_sent = sentence[:left] + [','] + sentence[left:right]
                front_move_indices = [i for i in range(sent_len+1) if i != (right-left)]
                end_move_indices = [i for i in range(sent_len+1) if i != left]

            else:
                inside_masked = sentence[:left] + [self.MASK] + sentence[right:]
                inside_reprs = self.get_output_reprs(inside_masked, None)

                front_move_sent = sentence[left:right] + [','] + sentence[:left] + [','] + sentence[right:]
                end_move_sent = sentence[:left] + [','] + sentence[right:] + [','] + sentence[left:right]
                front_move_indices = [i for i in range(sent_len+2) if i != (right-left) and i != (right+1)]
                end_move_indices = [i for i in range(sent_len+2) if i != left and i != (sent_len+1+left-right)]

            inside_reprs_nomask = torch.cat((inside_reprs[:left, :], inside_reprs[left+1:, :]), dim=0)
            inside_gold = torch.cat((gold_reprs[:left, :], gold_reprs[right:, :]), dim=0)
            inside_distortion = self.calc_distortion(inside_gold, inside_reprs_nomask)
            span_repr = self.get_output_reprs(sentence[left:right], None)
            span_distortion = self.calc_distortion(gold_reprs[left:right, :], span_repr)

            front_move_reprs = self.get_output_reprs(front_move_sent, None)[front_move_indices]
            end_move_reprs = self.get_output_reprs(end_move_sent, None)[end_move_indices]

            distortion_scores = []
            if left!=0:
                distortion_scores.append(self.calc_distortion(gold_reprs[:left, :], front_move_reprs[right-left:right, :]))
                distortion_scores.append(self.calc_distortion(gold_reprs[:left, :], end_move_reprs[:left, :]))
            if right!=sent_len:
                distortion_scores.append(self.calc_distortion(gold_reprs[right:, :], end_move_reprs[left:left+sent_len-right]))
                distortion_scores.append(self.calc_distortion(gold_reprs[right:, :], front_move_reprs[right:, :]))

            distortion_scores.append(self.calc_distortion(gold_reprs[left:right, :], front_move_reprs[:right-left, :]))
            distortion_scores.append(self.calc_distortion(gold_reprs[left:right, :], end_move_reprs[left+sent_len-right:, :]))

            distortion_scores.append(inside_distortion)
            distortion_scores.append(span_distortion)

            span_scores[left, right-1] = sum(distortion_scores) / len(distortion_scores)
        return span_scores

def convert_span_indices_to_str(span_set, sentence):
    str_set = set()
    for index_tuple in span_set:
        tokens = sentence[index_tuple[0]:index_tuple[1]+1]
        constituents = ' '.join(tokens)
        str_set.add(constituents)
    return str_set

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
    if "uncased" in config.model_name:
        config.lower = True
    else:
        config.lower = False
    test_ptb = load_ptb(test_ptb_path, lower=config.lower)
    model = PerturbParse(config.model_name, config.device, config.layer)

    labels = ['sbar', 'np', 'vp', 'pp', 'adjp', 'advp', 's']
    label_total = {'sbar':0, 'np':0, 'vp':0, 'pp':0, 'adjp':0, 'advp':0, 's':0}
    label_correct = {'sbar':0, 'np':0, 'vp':0, 'pp':0, 'adjp':0, 'advp':0, 's':0}

    corpus_f1 = [0.0, 0.0, 0.0]
    sent_f1 = []
    pred_all = []

    for i, example in enumerate(tqdm(test_ptb)):
        sent = example['sent']
        pred_spans = model(sent)
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
    parser.add_argument('--treebank_path', type=str, default="../data/ptb/ptb-dev-sample.txt")
    parser.add_argument('--model_name', type=str, default='roberta-base', help='bert-base-uncased, roberta-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--pred_tree_path', type=str, default='roberta-base-trail.txt')
    parser.add_argument('--layer', type=int, default=12)
    config = parser.parse_args()
    run(config)
    
if __name__ == '__main__':
    main()