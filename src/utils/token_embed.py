import torch
def embed_token_mean(words, tokenized, output_tensors):
    """ Handle words that are splitted into word pieces, 
    use wordpieces's mean token embedding as the word's embedding

    Args:
        words: a list of tokens 
        tokenized_sent: a list of tokens after tokenization
        output_tensors: BERT last hidden state [num_tokens, hidden_dim]
    """

    offset_mapping = tokenized.offsets    # [tokenized_length, 2]
    token_tensor = []
    sent_len = len(words.split(' '))
    sent_tensor = torch.zeros(sent_len, output_tensors.shape[1])
    tid = 0
    for i in range(len(offset_mapping)):
        if offset_mapping[i][0] == 0 and offset_mapping[i][1] == 0:     # special tokens
            continue
        else:
            token_tensor.append(output_tensors[i, :])
            if offset_mapping[i][1] == len(words):   # end of a word (sentence)
                sent_tensor[tid, :] = torch.stack(token_tensor).mean(axis=0)

            if offset_mapping[i][1] == offset_mapping[i+1][0]-1:   # end of a word
                sent_tensor[tid, :] = torch.stack(token_tensor).mean(axis=0)
                token_tensor = []
                tid += 1
    return sent_tensor
