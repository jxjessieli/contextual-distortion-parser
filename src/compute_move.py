import argparse
import torch
from transformers import RobertaTokenizer, RobertaModel, BertTokenizerFast, BertModel
from utils.trees_processing import load_ptb
from tqdm import tqdm
from utils.token_embed import embed_token_mean
import pickle
import numpy as np
from pathlib import Path

def prepare_all_spans(data):
    all_move_spans = []
    all_move_indices = []
    for i in range(len(data)):
        sentence = data[i]['sent']
        sent_len = len(sentence)
        for length in range(1, sent_len+1):
            for left in range(0, sent_len+1-length):
                right = left+length
                if left == 0:
                    front_move_sent =  sentence[left:right] + [','] + sentence[right:]
                    end_move_sent = sentence[right:] + [','] + sentence[left:right]
                    front_move_indices = [i for i in range(sent_len+1) if i != (right-left)]
                    end_move_indices = [i for i in range(sent_len+1) if i != (sent_len-right)]
                elif right == sent_len:
                    front_move_sent = sentence[left:right] + [','] + sentence[:left]
                    end_move_sent = sentence[:left] + [','] + sentence[left:right]
                    front_move_indices = [i for i in range(sent_len+1) if i != (right-left)]
                    end_move_indices = [i for i in range(sent_len+1) if i != left]
                else:
                    front_move_sent = sentence[left:right] + [','] + sentence[:left] + [','] + sentence[right:]
                    end_move_sent = sentence[:left] + [','] + sentence[right:] + [','] + sentence[left:right]
                    front_move_indices = [i for i in range(sent_len+2) if i != (right-left) and i != (right+1)]
                    end_move_indices = [i for i in range(sent_len+2) if i != left and i != (sent_len+1+left-right)]
                front_span = ' '.join(front_move_sent)
                end_span = ' '.join(end_move_sent)
                all_move_spans.append(front_span)
                all_move_spans.append(end_span)
                all_move_indices.append(front_move_indices)
                all_move_indices.append(end_move_indices)
    return all_move_spans, all_move_indices

def prepare_batched_spans(spans, indices, batch_size):
    num_batches = int(len(spans) / batch_size) + 1
    full_batched_spans = []
    full_batched_indices = []
    for i in range(num_batches):
        batch_spans = spans[i*batch_size:(i+1)*batch_size]
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        full_batched_spans.append(batch_spans)
        full_batched_indices.append(batch_indices)        
    return full_batched_spans, full_batched_indices

def prepare_batched_data(data, batch_size):
    num_batches = int(len(data) / batch_size) + 1
    full_batched_data = []
    for i in range(num_batches):
        batch_data = []
        for j in range(batch_size):
            if i*batch_size+j < len(data):
                batch_data.append(' '.join(data[i*batch_size+j]['sent']))
        full_batched_data.append(batch_data)
    return full_batched_data

def calc_distortion(gold_reprs, mask_reprs):
        """
        Args:
            gold_reprs: representations from the original sentence [num_tokens, hidden_dim]
            mask_reprs: representations from the masked sentence [num_tokens, hidden_dim]
        """
        num_tokens = gold_reprs.shape[0]
        distortion = torch.norm(gold_reprs - mask_reprs)**2 / num_tokens
        return distortion

def run(config):
    test_ptb_path = config.treebank_path

    test_ptb = load_ptb(test_ptb_path, lower=True)
    all_move_spans, all_move_indices = prepare_all_spans(test_ptb)
    full_bached_spans, full_bached_indices = prepare_batched_spans(all_move_spans, all_move_indices, config.batch_size)
    full_batched_ptb = prepare_batched_data(test_ptb, config.batch_size)

    # Initialize model
    if 'roberta-' in config.model_name:
        print('Roberta model: {}'.format(config.model_name))
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        model = RobertaModel.from_pretrained(config.model_name)
        MASK = '<mask>'
    elif 'bert-' in config.model_name:
        print('Bert model: {}'.format(config.model_name))
        tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
        model = BertModel.from_pretrained(config.model_name)
        MASK = '[MASK]'
    model.to(config.device)

    # Obtain representations
    full_data_reprs = []
    for i, batch_spans in enumerate(tqdm(full_bached_spans)):
        batch_indices = full_bached_indices[i]
        input = tokenizer(batch_spans, return_tensors='pt', padding=True, return_offsets_mapping=True).to(config.device)

        ## Layer outputs from PLM
        outputs_all = model(input_ids=input.data['input_ids'], attention_mask=input.data['attention_mask'], token_type_ids=input.data['token_type_ids'], output_hidden_states=True)
        output = outputs_all.hidden_states[config.layer].data.detach()

        for j in range(len(batch_spans)):
            output_reprs = embed_token_mean(batch_spans[j], input[j], output[j])
            full_data_reprs.append(output_reprs[batch_indices[j]].cpu())
        del outputs_all
        del output

    # Obtain gold representations for sentences
    full_gold_reprs = []
    for i, batch_data in enumerate(full_batched_ptb):
        input = tokenizer(batch_data, return_tensors='pt', padding=True, return_offsets_mapping=True).to(config.device)

        ## Layer outputs from PLM
        outputs_all = model(input_ids=input.data['input_ids'], attention_mask=input.data['attention_mask'], token_type_ids=input.data['token_type_ids'], output_hidden_states=True)
        output = outputs_all.hidden_states[config.layer].data.detach()

        for j in range(len(batch_data)):
            output_reprs = embed_token_mean(batch_data[j], input[j], output[j])
            full_gold_reprs.append(output_reprs.cpu())
        del outputs_all
        del output

    # Calculate distortion for each span in each sentence
    full_data_distortions = []
    curr_start_idx = 0
    for i in tqdm(range(len(test_ptb))):
        sent_len = len(test_ptb[i]['sent'])
        sent_repr_gold = full_gold_reprs[i]
        distortion_sent = torch.zeros(sent_len, sent_len)
        for length in range(1, sent_len+1):
            for left in range(0, sent_len+1-length):
                right = left + length
                if left == 0 and right == sent_len:
                    continue
                curr_span_idx_front = 2 * ((length-1) * (sent_len + sent_len - length + 2) / 2 + left)
                curr_span_idx_end = 2 * ((length-1) * (sent_len + sent_len - length + 2) / 2 + left) + 1
                front_move_sent_repr = full_data_reprs[int(curr_start_idx + curr_span_idx_front)]
                end_move_sent_repr = full_data_reprs[int(curr_start_idx + curr_span_idx_end)]
                if left != 0 and right != sent_len:
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[:left], front_move_sent_repr[right-left:right, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[:left], end_move_sent_repr[:left, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[right:, :], end_move_sent_repr[left:left+sent_len-right])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[right:, :], front_move_sent_repr[right:, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[left:right, :], front_move_sent_repr[:right-left, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[left:right, :], end_move_sent_repr[left+sent_len-right:, :])
                    distortion_sent[left][right-1] = distortion_sent[left][right-1] / 6
                elif left == 0:
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[right:, :], end_move_sent_repr[left:left+sent_len-right])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[right:, :], front_move_sent_repr[right:, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[left:right, :], front_move_sent_repr[:right-left, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[left:right, :], end_move_sent_repr[left+sent_len-right:, :])
                    distortion_sent[left][right-1] = distortion_sent[left][right-1] / 4
                elif right == sent_len:
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[:left], front_move_sent_repr[right-left:right, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[:left], end_move_sent_repr[:left, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[left:right, :], front_move_sent_repr[:right-left, :])
                    distortion_sent[left][right-1] += calc_distortion(sent_repr_gold[left:right, :], end_move_sent_repr[left+sent_len-right:, :])
                    distortion_sent[left][right-1] = distortion_sent[left][right-1] / 4
        full_data_distortions.append(distortion_sent)
        curr_start_idx += sent_len*(sent_len+1)

    # Write distortion scores to file
    Path("./scores/").mkdir(parents=True, exist_ok=True)
    with open(f'./scores/move_{config.score_path}_{config.layer}', 'wb') as out_file:
        pickle.dump(full_data_distortions, out_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-uncased', help='bert-base-uncased, roberta-base')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--score_path', type=str, default='bert-base-ptb')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--treebank_path', type=str, default="/../data/spmrl/ptb-dev.txt")
    config = parser.parse_args()
    run(config)
    
if __name__ == '__main__':
    main()