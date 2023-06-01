import os
DELETE = ('``', "''", '-NONE-', '.', '-none-')
PUNCT = (',', ';', ':', '--', '-', '.', '?', '!', '-LRB-', '-RRB-', '$', '#',
         '-LCB-', 'RCB-', "'", '-lrb-','-rrb-','-lcb-','-rcb-')

# from eval_for_comparison import get_actions, get_nonbinary_spans, get_stats
import itertools
import numpy as np
import nltk
import torch
import random

"""
Code is from https://github.com/harvardnlp/compound-pcfg
Their evaluation is slightly different from evalb so we use it for comparison.
"""

def all_binary_trees(n):
  #get all binary trees of length n
  def is_tree(tree, n):
    # shift = 0, reduce = 1
    if sum(tree) != n-1:
      return False
    stack = 0    
    for a in tree:
      if a == 0:
        stack += 1
      else:
        if stack < 2:
          return False
        stack -= 1
      if stack < 0:
        return False
    return True
  valid_tree = []
  num_shift = 0
  num_reduce = 0
  num_actions = 2*n - 1
  trees = map(list, itertools.product([0,1], repeat = num_actions-3))
  start = [0, 0] #first two actions are always shift
  end = [1] # last action is always reduce
  for tree in trees: 
    tree = start + tree + end
    if is_tree(tree, n):
      valid_tree.append(tree[::])
  return valid_tree

def get_tree(actions, sent = None, SHIFT = 0, REDUCE = 1):
  #input action and sent (lists), e.g. S S R S S R R, A B C D
  #output tree ((A B) (C D))
  stack = []
  pointer = 0
  if sent is None:
    sent = list(map(str, range((len(actions)+1) // 2)))
#  assert(len(actions) == 2*len(sent) - 1)
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      stack.append('(' + left + ' ' + right + ')')
  assert(len(stack) == 1)
  return stack[-1]
      
def get_depth(tree, SHIFT = 0, REDUCE = 1):
  stack = []
  depth = 0
  max = 0
  curr_max = 0
  for c in tree:
    if c == '(':
      curr_max += 1
      if curr_max > max:
        max = curr_max
    elif c == ')':
      curr_max -= 1
  assert(curr_max == 0)
  return max

def get_spans(actions, SHIFT = 0, REDUCE = 1):
  sent = list(range((len(actions)+1) // 2))
  spans = []
  pointer = 0
  stack = []
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      if isinstance(left, int):
        left = (left, None)
      if isinstance(right, int):
        right = (None, right)
      new_span = (left[0], right[1])
      spans.append(new_span)
      stack.append(new_span)
  return spans

def get_stats(span1, span2):
  tp = 0
  fp = 0
  fn = 0
  for span in span1:
    if span in span2:
      tp += 1
    else:
      fp += 1
  for span in span2:
    if span not in span1:
      fn += 1
  return tp, fp, fn

def update_stats(pred_span, gold_spans, stats):
  for gold_span, stat in zip(gold_spans, stats):
    tp, fp, fn = get_stats(pred_span, gold_span)
    stat[0] += tp
    stat[1] += fp
    stat[2] += fn

def get_f1(stats):
  f1s = []
  for stat in stats:
    prec = stat[0] / (stat[0] + stat[1]) if stat[0] + stat[1] > 0 else 0.
    recall = stat[0] / (stat[0] + stat[2]) if stat[0] + stat[2] > 0 else 0.
    f1 = 2*prec*recall / (prec + recall)*100 if prec+recall > 0 else 0.
    f1s.append(f1)
  return f1s

def get_random_tree(length, SHIFT = 0, REDUCE = 1):
  tree = [SHIFT, SHIFT]
  stack = ['', '']
  num_shift = 2
  while len(tree) < 2*length - 1:
    if len(stack) < 2:
      tree.append(SHIFT)
      stack.append('')
      num_shift += 1
    elif num_shift >= length:
      tree.append(REDUCE)
      stack.pop()
    else:
      if random.random() < 0.5:
        tree.append(SHIFT)
        stack.append('')
        num_shift += 1
      else:
        tree.append(REDUCE)
        stack.pop()
  return tree

def span_str(start = None, end = None):
  assert(start is not None or end is not None)
  if start is None:
    return ' '  + str(end) + ')'
  elif end is None:
    return '(' + str(start) + ' '
  else:
    return ' (' + str(start) + ' ' + str(end) + ') '    

def get_tree_from_binary_matrix(matrix, length):    
  sent = list(map(str, range(length)))
  n = len(sent)
  tree = {}
  for i in range(n):
    tree[i] = sent[i]
  for k in np.arange(1, n):
    for s in np.arange(n):
      t = s + k
      if t > n-1:
        break
      if matrix[s][t].item() == 1:
        span = '(' + tree[s] + ' ' + tree[t] + ')'
        tree[s] = span
        tree[t] = span
  return tree[0]
    

def get_nonbinary_spans(actions, SHIFT=0, REDUCE=1):
    spans = []
    tags = []
    stack = []
    pointer = 0
    binary_actions = []
    nonbinary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            nonbinary_actions.append(SHIFT)
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == 'NT(':
            # stack.append('(')
            stack.append(action[3:-1].split('-')[0])
        elif action == "REDUCE":
            nonbinary_actions.append(REDUCE)
            right = stack.pop()
            left = right
            n = 1
            # while stack[-1] is not '(':
            while type(stack[-1]) is tuple:
                left = stack.pop()
                n += 1
            span = (left[0], right[1])
            tag = stack.pop()
            if left[0] != right[1]:
                spans.append(span)
                tags.append(tag)
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)
                num_reduce += 1
        else:
            assert False
    assert (len(stack) == 1)
    assert (num_shift == num_reduce + 1)
    return spans, tags, binary_actions, nonbinary_actions

def get_nonbinary_tree(sent, tags, actions):
  pointer = 0
  tree = []
  for action in actions:
    if action[:2] == "NT":
      node_label = action[:-1].split("NT")[1]
      node_label = node_label.split("-")[0]
      tree.append(node_label)
    elif action == "REDUCE":
      tree.append(")")
    elif action == "SHIFT":
      leaf = "(" + tags[pointer] + " " + sent[pointer] + ")"
      pointer += 1
      tree.append(leaf)
    else:
      assert(False)
  assert(pointer == len(sent))
  return " ".join(tree).replace(" )", ")")

def build_tree(depth, sen):
  assert len(depth) == len(sen)

  if len(depth) == 1:
    parse_tree = sen[0]
  else:
    idx_max = np.argmax(depth)
    parse_tree = []
    if len(sen[:idx_max]) > 0:
      tree0 = build_tree(depth[:idx_max], sen[:idx_max])
      parse_tree.append(tree0)
    tree1 = sen[idx_max]
    if len(sen[idx_max + 1:]) > 0:
      tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
      tree1 = [tree1, tree2]
    if parse_tree == []:
      parse_tree = tree1
    else:
      parse_tree.append(tree1)
  return parse_tree

def get_brackets(tree, idx=0):
  brackets = set()
  if isinstance(tree, list) or isinstance(tree, nltk.Tree):
    for node in tree:
      node_brac, next_idx = get_brackets(node, idx)
      if next_idx - idx > 1:
        brackets.add((idx, next_idx))
        brackets.update(node_brac)
      idx = next_idx
    return brackets, idx
  else:
    return brackets, idx + 1

def get_nonbinary_spans_label(actions, SHIFT = 0, REDUCE = 1):
  spans = []
  stack = []
  pointer = 0
  binary_actions = []
  num_shift = 0
  num_reduce = 0
  for action in actions:
    # print(action, stack)
    if action == "SHIFT":
      stack.append((pointer, pointer))
      pointer += 1
      binary_actions.append(SHIFT)
      num_shift += 1
    elif action[:3] == 'NT(':
      label = "(" + action.split("(")[1][:-1]
      stack.append(label)
    elif action == "REDUCE":
      right = stack.pop()
      left = right
      n = 1
      while stack[-1][0] != '(':
        left = stack.pop()
        n += 1
      span = (left[0], right[1], stack[-1][1:])
      if left[0] != right[1]:
        spans.append(span)
      stack.pop()
      stack.append(span)
      while n > 1:
        n -= 1
        binary_actions.append(REDUCE)        
        num_reduce += 1
    else:
      assert False  
  assert(len(stack) == 1)
  assert(num_shift == num_reduce + 1)
  return spans, binary_actions

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')    

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('    
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]    

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx  
    return output_actions

  
def pcfg_compute_f1(tree1, tree2, length_cutoff = 10):
    corpus_f1 = [0., 0., 0.] 
    sent_f1 = [] 
    with torch.no_grad():
      for k, (tree1, tree2) in enumerate(zip(tree1, tree2)):
        tree1 = tree1.strip()
        try:
            action1 = get_actions(tree1)
        except Exception as e:
            print(e)
            print(tree1)
        tags1, sent1, sent_lower1 = get_tags_tokens_lowercase(tree1)
        if len(sent1) > length_cutoff or len(sent1) == 1:
            continue
        gold_span1, binary_actions1, nonbinary_actions1 = get_nonbinary_spans(action1)
        tree2 = tree2.strip()
        action2 = get_actions(tree2)
        tags2, sent2, sent_lower2 = get_tags_tokens_lowercase(tree2)
        gold_span2, binary_actions2, nonbinary_actions2 = get_nonbinary_spans(action2)
        pred_span_set = set(gold_span2[:-1]) #the last span in the list is always the
        gold_span_set = set(gold_span1[:-1]) #trival sent-level span so we ignore it
        tp, fp, fn = get_stats(pred_span_set, gold_span_set) 
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn
        # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py
        overlap = pred_span_set.intersection(gold_span_set)
        prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
        reca = float(len(overlap)) / (len(gold_span_set) + 1e-8)
        if len(gold_span_set) == 0:
            reca = 1.
            if len(pred_span_set) == 0:              
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.append(f1)
    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision", prec)
    print("Recall: ", recall)
    corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
    #print('Corpus F1: %.2f, Sentence F1: %.2f' %
    #      (corpus_f1*100, np.mean(np.array(sent_f1))*100))
    return np.mean(np.array(sent_f1))*100

class Tree(object):
    def __init__(self, label, children, word):
        self.label = label
        self.children = children
        self.word = word
        
    def __str__(self):
        return self.linearize()
    
    def linearize(self):
        if not self.children:
            return "({} {})".format(self.label, self.word)
        return "({} {})".format(self.label, " ".join(c.linearize() for c in self.children))

    def linearize_latex_labeled_fromparser(self):
        """
        Returns string for tikz-qtree, with nodes labeled by their score under the parser
        """
        if not self.children:
            return "{}".format(self.word)
        if self.label == 'X':
            prob = '1.00'
        else:
            prob = self.label[1:5]
        return "[ .{} {} ]".format(prob , " ".join(c.linearize_latex_labeled() for c in self.children))
    
    def linearize_latex_labeled_fromtreebank(self):
        """
        Returns string for tikz-qtree, use this method for trees from load_ptb
        """
        if not self.children:
            return "{}".format(self.word)
        return "[ .{} {} ]".format(self.label , " ".join(c.linearize_latex_labeled_2() for c in self.children))
    
    def linearize_latex_labeled_colored(self, gold_spans, start = 0):
        """
        Returns string for tikz-qtree. 
        Given this tree and a list of gold spans, color-codes the brackets:
            red - crossing, blue - correct, dashed blue - consistent with gold tree
        Note: the parser removes some punctuation. for the spans to align, both trees should have
        the same punctuation.
        """
        if not self.children:
            return "{}".format(self.word)
        if self.label == 'X':
            prob = '1.00'
        else:
            prob = self.label[1:5]
            
        position = start
        starts = []
        for c in self.children:
            starts.append(position)
            cspans = c.spans(start = position)
            position = cspans[0][1]
        span = (start, position)
        
        if span in gold_spans:
            edge = "\\edge[draw=blue];"
        elif crossing(span, gold_spans):
            edge = "\\edge[draw=red];"
        else:
            edge = "\\edge[dashed,draw=blue];"
        child_strings = [c.linearize_latex_labeled_colored(gold_spans, start) for c, start in zip(self.children, starts)]
        assert len(child_strings) == 2, 'Tree not binary'
        return "[ .{} {} {} {} {} ]".format(prob , edge, child_strings[0], edge, child_strings[1])
    
    def sent(self):
        if not self.children:
            return [self.word]
        return sum([c.sent() for c in self.children], [])
    
    def spans(self, start = 0):
        if not self.children:
            return [(start, start + 1)]
        span_list = []
        position = start
        for c in self.children:
            cspans = c.spans(start = position)
            span_list.extend(cspans)
            position = cspans[0][1]
        return [(start, position)] + span_list
        
    def spans_labels(self, start = 0):
        if not self.children:
            return [(start, start + 1, self.label)]
        span_list = []
        position = start
        for c in self.children:
            cspans = c.spans_labels(start = position)
            span_list.extend(cspans)
            position = cspans[0][1]
        return [(start, position, self.label)] + span_list
    
    def copy(self):
        return unlinearize(self.linearize())
 
def crossing(span, constraints):
    """
    False if span is consistent will all spans in constraints, True if at least
    one span is crossing.
    """
    i, j = span
    for (k,l) in constraints:
        if (i > k and i < l and j > l) or (i < k and j > k and j < l):
            return True
    return False
    
def unlinearize(string):
    """
    (TOP (S (NP (PRP He)) (VP (VBD was) (ADJP (JJ right))) (. .)))
    """
    tokens = string.replace("(", " ( ").replace(")", " ) ").split()
    
    def read_tree(start):
        if tokens[start + 2] != '(':
            return Tree(tokens[start+1], None, tokens[start+2]), start + 4
        i = start + 2
        children = []
        while tokens[i] != ')':
            tree, i = read_tree(i)
            children.append(tree)
        return Tree(tokens[start+1], children, None), i + 1
    
    tree, _ = read_tree(0)
    return tree

def collapse_unary_chains(tree):
    if tree.children:
        for i, child in enumerate(tree.children):
            while child.children and len(child.children) == 1:
                child = child.children[0]
            tree.children[i] = collapse_unary_chains(child)
    return tree

def standardize_punct(tree, nopunct): # tree cannot have unary chains
    if nopunct:
        delete_list = DELETE + PUNCT
    else:
        delete_list = DELETE
    if tree.children:
        tree.children = [c for c in tree.children if not (c.label in delete_list)]
        for c in tree.children:
            standardize_punct(c, nopunct)
    return tree
    
def remove_labels(tree):
    tree.label = 'X'
    if tree.children:
        for c in tree.children:
            remove_labels(c)
    return tree
        
def clean_empty(tree):
    """
    After removing punctuation, we might have non-leaves with 0 children.
    This function removes those nodes.
    """
    if tree.children:
        tree.children = [c for c in tree.children if c.word or len(c.children) > 0]
        for c in tree.children:
            clean_empty(c)
    return tree

def load_ptb(fname, nopunct = False, remove_len1 = False, lower = True):
    trees = []
    with open(fname) as file:
        for line in file:
            line = line.strip('\n')
            if lower:
                line = line.lower()
            tree = collapse_unary_chains(unlinearize(line))
            tree = standardize_punct(tree, nopunct)
            tree = collapse_unary_chains(clean_empty(tree))
            actions = get_actions(line)
            gold_spans, gold_tags, _, _ = get_nonbinary_spans(actions)

            if not remove_len1 or len([t for t in tree.sent() if t not in (DELETE+PUNCT)]) > 1:
                trees.append({'sent': tree.sent(), 'tree': tree, 'string': line, 'gold_spans': gold_spans, 'gold_tags': gold_tags})
    return trees
    
def detokenize(words):
    string = " ".join(words)
    string = string.replace('-LRB-', '(').replace('-RRB-', ')')
    string = string.replace('-LSB-', '[').replace('-RSB-', ']')
    string = string.replace('-LCB-', '{').replace('-RCB-', '}')
    return string.lower()

def get_leaves(tree):
    if not tree.children:
        return [tree]
    leaves = []
    for c in tree.children:
        leaves.extend(get_leaves(c))
    return leaves

def transfer_leaves(tree1, tree2):
    leaves1, leaves2 = get_leaves(tree1), get_leaves(tree2)
    for l1, l2 in zip(leaves1, leaves2):
        assert l1.word.lower() == l2.word.lower(), "{} =/= {}".format(l1.word, l2.word)
        l1.label = l2.label

def produce_right_branching(sent):
    if len(sent) == 1:
        return Tree(sent[0], None, sent[0]) # label, children, word
    return Tree('A', [produce_right_branching(sent[0:1]), produce_right_branching(sent[1:])], None) 

def binarize(tree): # non-destructive
    if not tree.children:
        return Tree(tree.label, None, tree.word)
    else:
        t = Tree(tree.label, [binarize(c) for c in tree.children], tree.word)
        while len(t.children) > 2:
            first, second = t.children[-2], t.children[-1]
            t.children = t.children[:-2]
            t.children.append(Tree("{}+{}".format(first.label, second.label), [first, second], None))
        return ts