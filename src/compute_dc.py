import argparse
import torch
from transformers import RobertaTokenizer, RobertaModel, BertTokenizerFast, BertModel
from utils.trees_processing import load_ptb
from tqdm import tqdm
from utils.token_embed import embed_token_mean
import pickle
import numpy as np
from pathlib import Path

## Note: outside means we mask the outside of the target span, which corresponds to decontextualization

def prepare_all_spans(data, MASK):
    all_spans = []
    for i in range(len(data)):
        sent = data[i]['sent']
        sent_len = len(sent)
        for length in range(1, sent_len+1):
            for left in range(0, sent_len+1-length):
                right = left + length
                if left == 0:
                    outside_masked = sent[left:right] + [MASK]
                elif right == sent_len:
                    outside_masked = [MASK] + sent[left:right]
                else:
                    outside_masked = [MASK] + sent[left:right] + [MASK]
                span = ' '.join(outside_masked)
                all_spans.append(span)
    return all_spans

def prepare_batched_spans(data, batch_size):
    num_batches = int(len(data) / batch_size) + 1
    full_batched_data = []
    for i in range(num_batches):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        full_batched_data.append(batch_data)
    return full_batched_data

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

    # Initialize model
    if 'roberta' in config.model_name:
        print('Roberta model: {}'.format(config.model_name))
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        model = RobertaModel.from_pretrained(config.model_name)
        MASK = '<mask>'
    else:
        print('Bert model: {}'.format(config.model_name))
        tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
        model = BertModel.from_pretrained(config.model_name)
        MASK = '[MASK]'
    model.to(config.device)

    test_ptb = load_ptb(test_ptb_path, lower=True)
    all_spans = prepare_all_spans(test_ptb, MASK)
    full_batched_spans = prepare_batched_spans(all_spans, config.batch_size)
    full_batched_ptb = prepare_batched_data(test_ptb, config.batch_size)

    # Obtain representations for spans
    full_span_reprs = []
    for i, batch_data in enumerate(tqdm(full_batched_spans)):
        input = tokenizer(batch_data, return_tensors='pt', padding=True, return_offsets_mapping=True).to(config.device)

        ## Layer outputs from PLM
        outputs_all = model(input_ids=input.data['input_ids'], attention_mask=input.data['attention_mask'], token_type_ids=input.data['token_type_ids'], output_hidden_states=True)
        output = outputs_all.hidden_states[config.layer].data.detach()

        for j in range(len(batch_data)):
            output_reprs = embed_token_mean(batch_data[j], input[j], output[j])
            full_span_reprs.append(output_reprs.cpu())
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
                curr_span_idx = (length-1) * (sent_len + sent_len - length + 2) / 2 + left
                outside_masked_repr = full_span_reprs[int(curr_start_idx + curr_span_idx)]
                if left == 0 and right == sent_len:
                    continue
                elif left == 0:
                    outside_repr = outside_masked_repr[left:right, :]
                elif right == sent_len:
                    outside_repr = outside_masked_repr[1:, :]
                else:
                    outside_repr = outside_masked_repr[1:-1, :]
                outside_gold = sent_repr_gold[left:right, :]
                distortion_sent[left][right-1] = calc_distortion(outside_gold, outside_repr)
        full_data_distortions.append(distortion_sent)
        curr_start_idx += sent_len*(sent_len+1)/2

    # Write distortion scores to file
    Path("./scores/").mkdir(parents=True, exist_ok=True)
    with open(f'./scores/dc_{config.score_path}_{config.layer}', 'wb') as out_file:
        pickle.dump(full_data_distortions, out_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-uncased', help='bert-base-uncased, roberta-base')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--score_path', type=str, default='bert-base-ptb')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--treebank_path', type=str, default="/../data/spmrl/ptb-dev.txt")
    config = parser.parse_args()
    run(config)
    
if __name__ == '__main__':
    main()