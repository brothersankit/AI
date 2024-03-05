# Import the required libraries.
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re, random
from copy import copy, deepcopy
import sys
from collections import Counter
import queue
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, f1_score
np.random.seed(21)
random.seed(21)


# In[65]:


import nltk

from nltk import FreqDist
from nltk import ngrams
from nltk.tag import pos_tag


# In[66]:


nltk.download('stopwords')
nltk.download('tagsets')


# In[67]:


nltk.download('tagsets')


# In[68]:


nltk.download('averaged_perceptron_tagger')


# In[69]:


from nltk.corpus import stopwords


# In[70]:


def text_label(data):
    text = []
    answer = []
    label = []
    sent_length = []
    for line in data:
        x = line.split(':', maxsplit=1)
        label.append(x[0])
        y = x[1].strip().split(' ',maxsplit=1)
        text.append(y[1].lower())
        answer.append(y[0])
    # remove punctuations
    ntext = [re.sub(r'([^\w\s]|[0-9])', ' ', line) for line in text]
    ntext = [re.sub(r'(\s+)', ' ', line) for line in ntext]
    return ntext, label


# In[71]:


def stopwords_remove(text):
    stop_word = set(stopwords.words('english'))
    stop_word.add('')
    tokens = [sent.split() for sent in text]
    text_n = [[w for w in words if w not in stop_word] for words in tokens]
    return text_n


# In[72]:


def ngram_topk(token_list, n, k, feat_dicts=None):
    list_ngram = [list(ngrams(sent, n)) for sent in token_list]
    if feat_dicts:
        ngrams_topk_dict, ngram_dict = feat_dicts
    else:
        all_ngrams = sum(list_ngram, [])
        freq_ = FreqDist(all_ngrams)
        freq_k = freq_.most_common(k)
        ngrams_topk_list =  [ngram_token for ngram_token, _ in freq_k]
        ngrams_topk_dict = {ngram_token:i for i, ngram_token in enumerate(ngrams_topk_list)}
        ngram_dict = {v: k for k, v in ngrams_topk_dict.items()}
    
    ngrams_freq_feat = []
    for tokens in list_ngram:
        ngram_token_freq = np.zeros(k, dtype = np.int32)
        for ngram_token in tokens:
            if ngram_token in ngrams_topk_dict.keys():
                ngram_token_freq[ ngrams_topk_dict[ngram_token] ]+=1
        ngrams_freq_feat.append(ngram_token_freq)
    
    return np.asarray(ngrams_freq_feat, dtype = np.int32), ngrams_topk_dict, ngram_dict, list_ngram


# In[73]:
os.chdir("C:\\RC\\Documents\\IIT Patna Remastered 2023-25\\Course\\1st Semester\\CS561 Artificial Intelligence\\Assignments\\Assignment 6 Decision Tree")
# print( os.getcwd() )
# Read the file line by line and clean the text of punctuation.
with open('train_data.txt', 'r',encoding='latin-1') as file:
    data = file.readlines()

X_train, Y_train = text_label(data)

with open('test_data.txt', 'r',encoding='latin-1') as file:
    data = file.readlines()

X_test, Y_test = text_label(data)


# In[74]:


def features_train(X_train, feats_to_use = ('len', 'uni' ,'bi' ,'tri' ,'pos')):
    X_train = [sent.split() for sent in X_train]
    features = []
    feat_dicts_list = []

    if 'len' in feats_to_use:
        X_train_sentlen = np.reshape(np.asarray([len(sent) for sent in X_train], dtype = np.int32),(-1,1))
        feat_dicts_list.append(None)
    else:
        X_train_sentlen = np.reshape(np.asarray([-1 for sent in X_train], dtype = np.int32),(-1,1))
    features.append(X_train_sentlen)
    
    if 'uni' in feats_to_use:
        X_train_unigram = ngram_topk(X_train, 1, 500)
        features.append(X_train_unigram[0])
        feat_dicts_list.append(X_train_unigram[1:-1])
    if 'bi' in feats_to_use:
        X_train_bigram = ngram_topk(X_train, 2, 300)
        features.append(X_train_bigram[0])
        feat_dicts_list.append(X_train_bigram[1:-1])
    if 'tri' in feats_to_use:
        X_train_trigram = ngram_topk(X_train, 3, 200)
        features.append(X_train_trigram[0])
        feat_dicts_list.append(X_train_trigram[1:-1])
    if 'pos' in feats_to_use:
        X_train_pos = [pos_tag(tokens) for tokens in X_train]
        X_train_pos_unigram = ngram_topk(X_train_pos, 1, 500)
        features.append(X_train_pos_unigram[0])
        feat_dicts_list.append(X_train_pos_unigram[1:-1])

    X_train_feats = np.concatenate( features, axis=1 )
    
    return X_train_feats, feat_dicts_list


# In[75]:


def get_features_test(X_test, feat_dicts_list, feats_to_use = ('len', 'uni' ,'bi' ,'tri' ,'pos')):
    X_test = [sent.split() for sent in X_test]
    features = []
    feat_dicts_list_idx = 0
    assert len(feat_dicts_list) == len(feats_to_use)
    if 'len' in feats_to_use:
        X_test_sentlen = np.reshape(np.asarray([len(sent) for sent in X_test], dtype = np.int32),(-1,1))
        feat_dicts_list_idx += 1
    else:
        X_test_sentlen = np.reshape(np.asarray([-1 for sent in X_test], dtype = np.int32),(-1,1))
    features.append(X_test_sentlen)
    
    if 'uni' in feats_to_use:
        X_test_unigram = ngram_topk(X_test, 1, 500, feat_dicts_list[feat_dicts_list_idx])
        features.append(X_test_unigram[0])
        feat_dicts_list_idx += 1
    if 'bi' in feats_to_use:
        X_test_bigram = ngram_topk(X_test, 2, 300, feat_dicts_list[feat_dicts_list_idx])
        features.append(X_test_bigram[0])
        feat_dicts_list_idx += 1
    if 'tri' in feats_to_use:
        X_test_trigram = ngram_topk(X_test, 3, 200, feat_dicts_list[feat_dicts_list_idx])
        features.append(X_test_trigram[0])
        feat_dicts_list_idx += 1
    if 'pos' in feats_to_use:
        X_test_pos = [pos_tag(tokens) for tokens in X_test]
        X_test_pos_unigram = ngram_topk(X_test_pos, 1, 500, feat_dicts_list[feat_dicts_list_idx])
        features.append(X_test_pos_unigram[0])
        feat_dicts_list_idx += 1
    
    assert len(feat_dicts_list) == feat_dicts_list_idx
    X_test_feats = np.concatenate( features, axis=1 )
    return X_test_feats


# In[76]:


X_train_feats, feat_dicts_list = features_train(X_train, feats_to_use = ('len', 'uni' ,'bi' ,'tri' ,'pos'))


# In[77]:


X_test_feats = get_features_test(X_test, feat_dicts_list, feats_to_use = ('len', 'uni' ,'bi' ,'tri' ,'pos'))


# In[78]:


label2idx = {lab: i for i, lab in enumerate(set(Y_train))}
idx2label = {i: lab for lab, i in label2idx.items()}
print(label2idx)


# In[79]:


Y_train_idx = np.asarray([label2idx[lab] for lab in Y_train], dtype=np.int32)#Gain type: gini)
Y_test_idx = np.asarray([label2idx[lab] for lab in Y_test], dtype=np.int32)


# In[80]:


class Node():
    cnt = 0
    def __init__(self, ):
        self.leaf = False
        self.majority_class = None
        self.attribute_index = None
        self.children = dict() # key: attribute_value, value: child_node
        self.sent_len_split_val = None # Used at inference time, if attribute_index is 0
        self.id = Node.cnt
        Node.cnt+=1
    
    def __str__(self,):
        return 'ID: {}  isLeaf: {} majority: {} split_idx: {} split_val = {}'.format(self.id, 
                                                                                    self.leaf, 
                                                                                    self.majority_class, 
                                                                                    self.attribute_index, 
                                                                                    list(self.children.keys())
                                                                                   )
    def __repr__(self):
        return str(self)
    
    def traverse_print(self,):
        print(self)
        for _, child in self.children:
              child.traverse_print()

    @classmethod
    def reset_cnt(cls,):
        cls.cnt = 0


# In[81]:


class DecisionTree():
    
    # score has to be from 'entropy', 'gini', 'misclassification'
    def __init__(self, score='entropy'):
        score_functions = {'entropy': (DecisionTree.compute_entropy, DecisionTree.get_gain_entropy),
           'gini': (DecisionTree.compute_gini, DecisionTree.get_gain_gini),
           'misclassification': (DecisionTree.compute_misclassification, DecisionTree.get_gain_misclassification)}
        
        self.root = None
        assert score in score_functions.keys()
        self.score = score
        self.compute_score = score_functions[score][0]
        self.get_gain = score_functions[score][1]
        return
    
    @staticmethod
    def compute_entropy(labels):
        entropy = 0.0
        totSamples = len(labels)
        labelSet = set(labels.reshape(-1))
        for label in labelSet:
            prob = np.sum(labels == label) / totSamples
            if prob > 1e-12:
                entropy -= np.log(prob) * prob
        
        return entropy
    
    @staticmethod
    def get_gain_entropy(parent_info, data_i, labels):
        attr_split_info = 0
        attr_count = dict()
        for attr_val in set(data_i.reshape(-1)):
            ids = np.where(data_i == attr_val)[0]
            attr_count[attr_val] = len(ids)
            attr_split_info += attr_count[attr_val] * DecisionTree.compute_entropy(labels[ids])
        attr_gain = parent_info - attr_split_info
        attr_gain_ratio = DecisionTree.compute_dict_entropy(attr_count) * attr_gain
        return attr_gain, attr_gain_ratio, attr_count.keys()
    
    @staticmethod
    def compute_dict_entropy(attr_count):
        entropy = 0
        totSamples = sum(attr_count.values())
       
        labelSet = attr_count.keys()
        for label in labelSet:
            prob = attr_count[label] / totSamples
            if prob > 1e-12:
                entropy -= np.log(prob) * prob
        return entropy
    
    @staticmethod
    def compute_gini(labels):
        prob_sq = 0.0
        totSamples = len(labels)
        labelSet = set(labels.reshape(-1))
        for label in labelSet:
            prob = np.sum(labels == label) / totSamples
            if prob > 1e-12:
                prob_sq += prob*prob
        return 1-prob_sq
        
    
    @staticmethod
    def get_gain_gini(parent_info, data_i, labels):
        attr_split_info = 0
        attr_count = dict()
        for attr_val in set(data_i.reshape(-1)):
            ids = np.where(data_i == attr_val)[0]
            attr_count[attr_val] = len(ids)
            attr_split_info += attr_count[attr_val] * DecisionTree.compute_gini(labels[ids])
        attr_split_info /= data_i.shape[0]
        
        attr_gain = parent_info - attr_split_info
        #attr_gain_ratio = DecisionTree.compute_dict_entropy(attr_count) * attr_gain
        return attr_gain, attr_gain, attr_count.keys()

    @staticmethod
    def compute_misclassification(labels):
        max_prob = -1
        totSamples = len(labels)
        labelSet = set(labels.reshape(-1))
        for label in labelSet:
            prob = np.sum(labels == label) / totSamples
            if prob > max_prob:
                max_prob = prob
        
        return 1-max_prob
    
    @staticmethod
    def get_gain_misclassification(parent_info, data_i, labels):
        attr_split_info = -1
        attr_count = dict()
        for attr_val in set(data_i.reshape(-1)):
            ids = np.where(data_i == attr_val)[0]
            attr_count[attr_val] = len(ids)
            attr_split_info = attr_count[attr_val] * DecisionTree.compute_misclassification(labels[ids])
        attr_split_info /= data_i.shape[0]
        attr_gain = parent_info - attr_split_info
        return attr_gain, attr_gain, attr_count.keys()
    
    def split_node(self, parent, data, labels, used_attr_index):
        num_instances = data.shape[0]
        parent_info = self.compute_score(labels) * num_instances
        parent.majority_class = Counter(labels.reshape(-1)).most_common(1)[0][0]
        
        if parent_info == 0 :
            parent.leaf = True
        
        best_attr_index = None
        best_info_gain = -float('inf')
        best_gain_ratio = -float('inf')
        best_attr_keys = None
#         sent length case special
#         attr_split_info = 0
#         attr_count = dict()
        sent_len_split_val = stats.mode(data[:, 0])[0][0]
        le_ids = np.where(data[:, 0] <= sent_len_split_val)[0]
        gt_ids = np.where(data[:, 0] > sent_len_split_val)[0]
        data_0 = np.zeros(data.shape[0], dtype=np.int32)
        data_0[gt_ids] = 1
#         attr_count[0] = le_ids.shape[0]
#         attr_count[1] = gt_ids.shape[0]
#         attr_split_info = (attr_count[0] * self.compute_entropy(labels[le_ids])) + (attr_count[1] * self.compute_entropy(labels[gt_ids]) )    
#         attr_gain = parent_info - attr_split_info
        attr_gain, attr_gain_ratio, attr_count_keys = self.get_gain(parent_info, data_0, labels)
#         attr_gain_ratio = self.compute_dict_entropy(attr_count) * attr_gain
        if best_gain_ratio < attr_gain_ratio and  attr_gain_ratio > 0 :
                best_attr_index = 0
                best_info_gain = attr_gain
                best_gain_ratio = attr_gain_ratio
                best_attr_keys = attr_count_keys
        
        # during ablation, sentence length can be initialized to all zeros this will prevent splittiung in sent dimension/.
        for i in range(1, data.shape[1]): # starts from 1 as zero is sentence length (always.) .
            if i in used_attr_index:
                continue
            attr_gain, attr_gain_ratio, attr_count_keys = self.get_gain(parent_info, data[:, i], labels)
            if best_gain_ratio < attr_gain_ratio:
                best_attr_index = i
                best_info_gain = attr_gain
                best_gain_ratio = attr_gain_ratio
                best_attr_keys = attr_count_keys
        if best_gain_ratio <= 0 or len(best_attr_keys) == 1 :
            parent.leaf = True
            return [] # TO Check    
        else:
            parent.attribute_index =  best_attr_index
            parent.children = { i: Node() for i in best_attr_keys}
            to_return = []
            if best_attr_index != 0:
                used_attr_index.append(best_attr_index)
                for i in best_attr_keys:
                    inds = np.where(data[:, best_attr_index] == i)[0]
                    to_return.append( (parent.children[i], data[inds], labels[inds], used_attr_index) )
            else:
                parent.sent_len_split_val = sent_len_split_val
#                 print(len(best_attr_keys))
                for i in best_attr_keys:
                    inds = np.where(data_0 == i)[0]
                    to_return.append( (parent.children[i], data[inds], labels[inds], used_attr_index) )
                    
#                 to_return.append( (parent.children[0], data[le_ids], labels[le_ids], used_attr_index) )
#                 to_return.append( (parent.children[1], data[gt_ids], labels[gt_ids], used_attr_index) )
            return to_return
    
    def build_tree(self, data, labels):
        traversal_q = queue.Queue()
        root = Node()
        self.root = root
        traversal_q.put_nowait( (root, data, labels, [] ))
#         cent = 0
        while not traversal_q.empty():
            node_to_split = traversal_q.get_nowait()
            child_nodes = self.split_node(*node_to_split)
            for child in child_nodes:
                traversal_q.put_nowait(child)
#             if Node.cnt % 100 == 0:
#                 print(Node.cnt)
#                 cent+=1
        return root
    
    def split_infer(self, node, data, data_indices):
        if node.leaf:
            return (True, data_indices, np.zeros( (data.shape[0]), dtype = np.int32) + node.majority_class)
        else:
            to_queue = []
            if(node.attribute_index == 0):
                left_idx = np.where(data[:,0] <= node.sent_len_split_val)[0]
                right_idx = np.where(data[:,0] > node.sent_len_split_val)[0]
                to_queue.append( (node.children[0], data[left_idx], data_indices[left_idx]) )
                to_queue.append( (node.children[1], data[right_idx], data_indices[right_idx]) )
                return (False, to_queue)
            else:
                for i in node.children.keys():
                    split_inds = np.where( data[:, node.attribute_index]  == i)[0]
                    if len(split_inds) > 0:
                        to_queue.append( (node.children[i], data[split_inds], data_indices[split_inds]) )
                return (False, to_queue)
    
    def split_infer_depth(self, node, data, data_indices,depth):
        if node.leaf or depth >=10:
            return (True, data_indices, np.zeros( (data.shape[0]), dtype = np.int32) + node.majority_class)
        else:
            to_queue = []
            if(node.attribute_index == 0):
                left_idx = np.where(data[:,0] <= node.sent_len_split_val)[0]
                right_idx = np.where(data[:,0] > node.sent_len_split_val)[0]
                to_queue.append( (node.children[0], data[left_idx], data_indices[left_idx],depth+1) )
                to_queue.append( (node.children[1], data[right_idx], data_indices[right_idx], depth+1) )
                return (False, to_queue)
            else:
                for i in node.children.keys():
                    split_inds = np.where( data[:, node.attribute_index]  == i)[0]
                    if len(split_inds) > 0:
                        to_queue.append( (node.children[i], data[split_inds], data_indices[split_inds],depth+1) )
                return (False, to_queue)
    
    def get_labels(self, data):
        root = self.root
        data_idx = np.arange(data.shape[0], dtype = np.int32)
        labels = np.zeros( (data.shape[0]), dtype = np.int32) + -1
        traversal_q = queue.Queue()
        traversal_q.put_nowait( (root, data, data_idx ))
        while not traversal_q.empty():
            node_to_split = traversal_q.get_nowait()
            split_return = self.split_infer(*node_to_split)
            if split_return[0]:
                labels[split_return[1]] = split_return[2]
            else:
                for child in split_return[1]:
                    traversal_q.put_nowait(child)
        return labels
    
    def get_labels_depth(self, data):
        root = self.root
        data_idx = np.arange(data.shape[0], dtype = np.int32)
        labels = np.zeros( (data.shape[0]), dtype = np.int32) + -1
        traversal_q = queue.Queue()
        traversal_q.put_nowait( (root, data, data_idx ,0))
        while not traversal_q.empty():
            node_to_split = traversal_q.get_nowait()
            split_return = self.split_infer_depth(*node_to_split)
            if split_return[0]:
                labels[split_return[1]] = split_return[2]
            else:
                for child in split_return[1]:
                    traversal_q.put_nowait(child)
        return labels


# In[82]:


# Binarize data
X_train_feats_bin = deepcopy(X_train_feats)
X_train_feats_bin[:, 1:] = (X_train_feats[:, 1:] > 0).astype(np.int32)

X_test_feats_bin = deepcopy(X_test_feats)
X_test_feats_bin[:, 1:] = (X_test_feats[:, 1:] > 0).astype(np.int32)
dtree = DecisionTree('gini')
root = dtree.build_tree(data=X_train_feats_bin, labels=Y_train_idx)


# In[83]:


root


# In[84]:


Node.cnt


# In[85]:


dtree = DecisionTree('entropy')
root = dtree.build_tree(data=X_train_feats_bin, labels=Y_train_idx)


# In[86]:


dtree.root


# In[87]:


y_pred_test = dtree.get_labels(data=X_test_feats_bin)


# In[88]:


def get_scores(Y_test_idx, y_pred_test, average='weighted'):
    acc = (y_pred_test == Y_test_idx).mean()
    prec, rec, fscore, _ = precision_recall_fscore_support(Y_test_idx, y_pred_test, average=average)
    return acc, prec, rec, fscore


# In[89]:


print('Acc: {}, Prec: {}, Rec: {}, Fscore: {}'.format(*get_scores(Y_test_idx, y_pred_test)))


# In[90]:


precision_recall_fscore_support(Y_test_idx, y_pred_test, average='macro')


# In[91]:


all_features = {'len', 'uni' ,'bi' ,'tri' ,'pos'}
dtree_list = dict()
scores_list = dict()

for feat_to_drop in all_features:
    feats_to_use = frozenset(all_features - {feat_to_drop})
    X_train_feats, feat_dicts_list = features_train(X_train, feats_to_use = feats_to_use)
    X_test_feats = get_features_test(X_test, feat_dicts_list, feats_to_use = feats_to_use)
    X_train_feats_bin = deepcopy(X_train_feats)
    X_train_feats_bin[:, 1:] = (X_train_feats[:, 1:] > 0).astype(np.int32)
    
    X_test_feats_bin = deepcopy(X_test_feats)
    X_test_feats_bin[:, 1:] = (X_test_feats[:, 1:] > 0).astype(np.int32)
    dtree = DecisionTree()
    _ = dtree.build_tree(data=X_train_feats_bin, labels=Y_train_idx)
    dtree_list[feats_to_use] = dtree
    y_pred_test = dtree.get_labels(data=X_test_feats_bin)
    all_scores = get_scores(Y_test_idx, y_pred_test)
    scores_list[feats_to_use] = all_scores
    print('Features: {}, Missing Feature: {}'.format(feats_to_use, feat_to_drop))
    print('Acc: {}, Prec: {}, Rec: {}, Fscore: {}'.format(*all_scores))
    print()


# In[92]:


all_features = {'len', 'uni' ,'bi' ,'tri' ,'pos'}

feats_to_use = frozenset(all_features - {'uni', 'bi', 'tri'})
X_train_feats, feat_dicts_list = features_train(X_train, feats_to_use = feats_to_use)
X_test_feats = get_features_test(X_test, feat_dicts_list, feats_to_use = feats_to_use)
X_train_feats_bin = deepcopy(X_train_feats)
X_train_feats_bin[:, 1:] = (X_train_feats[:, 1:] > 0).astype(np.int32)

X_test_feats_bin = deepcopy(X_test_feats)
X_test_feats_bin[:, 1:] = (X_test_feats[:, 1:] > 0).astype(np.int32)
dtree = DecisionTree()
_ = dtree.build_tree(data=X_train_feats_bin, labels=Y_train_idx)
dtree_list[feats_to_use] = dtree
y_pred_test = dtree.get_labels(data=X_test_feats_bin)
all_scores = get_scores(Y_test_idx, y_pred_test)
scores_list[feats_to_use] = all_scores
print('Features: {}, Missing Feature: {}'.format(feats_to_use, feat_to_drop))
print('Acc: {}, Prec: {}, Rec: {}, Fscore: {}'.format(*all_scores))
print()


# In[93]:


all_features = {'len', 'uni' ,'bi' ,'tri' ,'pos'}
dtree_list = dict()
scores_dict = dict()
preds_dict = dict()
gain_types = ['entropy', 'gini', 'misclassification']

for gain_type in gain_types:
    X_train_feats, feat_dicts_list = features_train(X_train, feats_to_use = all_features)
    X_test_feats = get_features_test(X_test, feat_dicts_list, feats_to_use = all_features)
    X_train_feats_bin = deepcopy(X_train_feats)
    X_train_feats_bin[:, 1:] = (X_train_feats[:, 1:] > 0).astype(np.int32)
    X_test_feats_bin = deepcopy(X_test_feats)
    X_test_feats_bin[:, 1:] = (X_test_feats[:, 1:] > 0).astype(np.int32)
    dtree = DecisionTree(score=gain_type)
    _ = dtree.build_tree(data=X_train_feats_bin, labels=Y_train_idx)
    dtree_list[gain_type] = dtree
    y_pred_test = dtree.get_labels(data=X_test_feats_bin)
    all_scores = get_scores(Y_test_idx, y_pred_test)
    scores_dict[gain_type] = all_scores
    preds_dict[gain_type] = y_pred_test
    print('Gain type: {}'.format(gain_type))
    print('Acc: {}, Prec: {}, Rec: {}, Fscore: {}'.format(*all_scores))
    print()


# In[94]:


# Compute per-class f1 scores, and print the best model types for each class
f1_scores = np.zeros((len(gain_types), len(label2idx)))
for idx, gain_type in enumerate(gain_types):
    y_pred = preds_dict[gain_type]
    f1_scores[idx] = get_scores(preds_dict[gain_type], Y_test_idx, None)[-1]

for label in label2idx.keys():
    best_method = np.argmax(f1_scores[:, label2idx[label]])
    print("Best gain type is {} for class {}, {}-only-f1-score: {}".format(gain_types[best_method], label, label, f1_scores[best_method, label2idx[label]])) 


# In[95]:


error_analysis = [np.where((preds_dict['entropy'] != Y_test_idx) & ((preds_dict['gini'] == Y_test_idx) | (preds_dict['misclassification'] == Y_test_idx)))[0],
np.where(((preds_dict['gini'] != Y_test_idx) & ((preds_dict['entropy'] == Y_test_idx) | (preds_dict['misclassification'] == Y_test_idx))))[0],
np.where(((preds_dict['entropy'] == Y_test_idx) | (preds_dict['gini'] == Y_test_idx)) & (preds_dict['misclassification'] != Y_test_idx))[0]]


# In[96]:


for idx, gain_type in enumerate(gain_types):
    print("Examples that {} got wrong,and everything else got right:".format(gain_type))
    perc = len(error_analysis[idx]) * 100.0 / np.sum(preds_dict[gain_type] != Y_test_idx)
    print("Percentage of corrected examples: {}%\n".format(perc))
    for idx in error_analysis[idx][:10]:
        print(data[idx].strip())
    print('---\n\n')


# In[ ]:




