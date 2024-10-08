import numpy as np
import torch
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re,json
sys.path.append("/home/leon/project_vincy/syn_bert/third_party/processors")
from tree import *
from constants import *
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # training nodes are training docs, no initial features
    # print("x: ", x)
    # test nodes are training docs, no initial features
    # print("tx: ", tx)
    # both labeled and unlabeled training instances are training docs and words
    # print("allx: ", allx)
    # training labels are training doc labels
    # print("y: ", y)
    # test labels are test doc labels
    # print("ty: ", ty)
    # ally are labels for labels for allx, some will not have labels, i.e., all 0
    # print("ally: \n")
    # for i in ally:
    # if(sum(i) == 0):
    # print(i)
    # graph edge weight is the word co-occurence or doc word frequency
    # no need to build map, directly build csr_matrix
    # print('graph : ', graph)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj_text','adj_gaze','adj_fuse']
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj_text', 'adj_gaze']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str,dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    # x, y, tx, ty, allx, ally,adj_text,adj_gaze,adj_fuse = tuple(objects)
    x, y, tx, ty, allx, ally, adj_text, adj_gaze = tuple(objects)
    # print(x.shape, y.shape,gaze.shape, tx.shape, ty.shape, tgaze.shape,allx.shape, ally.shape,allgaze.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    # print(allgaze.shape,gaze.shape)
    # gazes = np.vstack((allgaze,tgaze))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}/{}.train.index".format(dataset_str,dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # gaze_train = np.zeros(labels.shape)
    # gaze_val = np.zeros(labels.shape)
    # gaze_test = np.zeros(labels.shape)
    # gaze_train[train_mask, :] = gazes[train_mask, :]
    # gaze_val[val_mask, :] = gazes[val_mask, :]
    # gaze_test[test_mask, :] = gazes[test_mask, :]

    adj_text = adj_text + adj_text.T.multiply(adj_text.T > adj_text) - adj_text.multiply(adj_text.T > adj_text)
    adj_gaze = adj_gaze + adj_gaze.T.multiply(adj_gaze.T > adj_gaze) - adj_gaze.multiply(adj_gaze.T > adj_gaze)
    # adj_fuse = adj_fuse + adj_fuse.T.multiply(adj_fuse.T > adj_fuse) - adj_fuse.multiply(adj_fuse.T > adj_fuse)

    # return adj_text,adj_gaze,adj_fuse, features, y_train, y_val, y_test,\
    #        train_mask, val_mask, test_mask, train_size, test_size
    return adj_text,adj_gaze, features, y_train, y_val, y_test,\
           train_mask, val_mask, test_mask, train_size, test_size

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# ------------------------------------------------------
class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, words,
                 heads=None, dep_tags=None, pos_tags=None,
                 FFDs=None,GDs=None,GPTs=None,TRTs=None,nFixs=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          words: list. The words of the sequence.
          labels: (Optional) list. The labels for each word of the sequence. This should be
          specified for train and dev examples, but not for test examples.
        """
        self.words = words
        self.heads = heads
        self.dep_tags = dep_tags
        self.pos_tags = pos_tags
        self.FFDs = FFDs
        self.GDs = GDs
        self.GPTs = GPTs
        self.TRTs = TRTs
        self.nFixs = nFixs

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            dep_tag_ids=None,
            pos_tag_ids=None,
            root=None,
            heads=None,
            depths=None,
            trunc_token_ids=None,
            sep_token_indices=None,
            FFDs=None,
            GDs=None,
            GPTs=None,
            TRTs=None,
            nFixs=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.dep_tag_ids = dep_tag_ids
        self.pos_tag_ids = pos_tag_ids
        # self.langs = langs
        self.root = root
        self.heads = heads
        self.trunc_token_ids = trunc_token_ids
        self.sep_token_indices = sep_token_indices
        self.depths = depths
        self.FFDs = FFDs,
        self.GDs = GDs,
        self.GPTs = GPTs,
        self.TRTs = TRTs,
        self.nFixs = nFixs,

def process_sentence(
        token_list, head_list, dep_tag_list,
        pos_tag_list, tokenizer,
        FFD_list,GD_list,GPT_list,TRT_list,nFix_list
):
    """
    When a token gets split into multiple word pieces,
    we make all the pieces (except the first) children of the first piece.
    However, only the first piece acts as the node that contains
    the dependent tokens as the children.
    """
    assert len(token_list) == len(head_list)  == \
           len(dep_tag_list) == len(pos_tag_list)
    # print('----token list-----')
    # print(token_list)
    text_tokens = []
    text_deptags = []
    text_postags = []
    text_FFDs, text_GDs, text_GPTs, text_TRTs, text_nFixs = [], [], [], [], []
    # My name is Wa ##si Ah ##mad
    # 0  1    2  3  3    4  4
    sub_tok_to_orig_index = []
    # My name is Wa ##si Ah ##mad
    # 0  1    2  3       5
    old_index_to_new_index = []
    # My name is Wa ##si Ah ##mad
    # 1  1    1  1  0    1  0
    first_wpiece_indicator = []
    offset = 0

    for i, token in enumerate(token_list):
        # print(token)
        word_tokens = tokenizer.tokenize(token)

        # print('token:',token,' word_tokens:',word_tokens)
        if len(token) != 0 and len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        old_index_to_new_index.append(offset)  # word piece index
        offset += len(word_tokens)
        for j, word_token in enumerate(word_tokens):
            first_wpiece_indicator += [1] if j == 0 else [0]
            # labels += [label_map[label]] if j == 0 else [pad_token_label_id]
            text_tokens.append(word_token)
            sub_tok_to_orig_index.append(i)
            text_deptags.append(dep_tag_list[i])
            text_postags.append(pos_tag_list[i])
            text_FFDs.append(FFD_list[i])
            text_GDs.append(GD_list[i])
            text_GPTs.append(GPT_list[i])
            text_TRTs.append(TRT_list[i])
            text_nFixs.append(nFix_list[i])
        # print('labels:',labels)
        # print('first_wpiece_indicator:',first_wpiece_indicator)
        # print('text_tokens:',text_tokens)
        # print('sub_tok_to_orig_index:',sub_tok_to_orig_index)
        # print('text_deptags:',text_deptags)
        # print('text_postags:',text_postags)
        # print('text_FFDs:',text_FFDs)

    # print('----------------------------------------')
    assert len(text_tokens) == len(sub_tok_to_orig_index), \
        "{} != {}".format(len(text_tokens), len(sub_tok_to_orig_index))
    assert len(text_tokens) == len(first_wpiece_indicator)

    text_heads = []
    head_idx = -1
    assert max(head_list) <= len(head_list), (max(head_list), len(head_list))
    # iterating over the word pieces to adjust heads
    for i, orig_idx in enumerate(sub_tok_to_orig_index):
        # orig_idx: index of the original word (the word-piece belong to)
        head = head_list[orig_idx]
        if head == 0:  # root
            # if root word is split into multiple pieces,
            # we make the first piece as the root node
            # and all the other word pieces as the child of the root node
            if head_idx == -1:
                head_idx = i + 1
                text_heads.append(0)
            else:
                text_heads.append(head_idx)
        else:
            if first_wpiece_indicator[i] == 1:
                # head indices start from 1, so subtracting 1
                head = old_index_to_new_index[head - 1]
                text_heads.append(head + 1)
            else:
                # word-piece of a token (except the first)
                # so, we make the first piece the parent of all other word pieces
                head = old_index_to_new_index[orig_idx]
                text_heads.append(head + 1)
    # print('text_heads:',text_heads)
    assert len(text_tokens) == len(text_heads), \
        "{} != {}".format(len(text_tokens), len(text_heads))
    # exit(0)
    return text_tokens, text_heads,  text_deptags, text_postags,\
           text_FFDs, text_GDs, text_GPTs, text_TRTs, text_nFixs

def read_examples_from_file(file_path):
    examples = []
    name_list = ['words', 'heads', 'dep_tags', 'pos_tags',
                 'FFDs', 'GDs', 'GPTs', 'TRTs', 'nFixs']
    name_dict = {}
    for i in name_list:
        name_dict[i] = []
    num=0
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            ex = json.loads(line)
            num+=1
            name_dict['words'].append(ex['tokens'])
            name_dict['heads'].append(ex['head'])
            name_dict['dep_tags'].append([tag.split(':')[0] if ':' in tag else tag \
                                          for tag in ex['deptag']])
            name_dict['pos_tags'].append(ex['postag'])
            new_ex = {}
            new_ex['FFD'], new_ex['GD'], new_ex['GPT'], new_ex['TRT'], new_ex['nFix'] = [], [], [], [], []
            for i in range(len(ex['FFD'])):
                new_ex['FFD'].append(int(float(ex['FFD'][i])))
                new_ex['GD'].append(int(float(ex['GD'][i])))
                new_ex['GPT'].append(int(float(ex['GPT'][i])))
                new_ex['TRT'].append(int(float(ex['TRT'][i])))
                new_ex['nFix'].append(int(float(ex['nFix'][i])))
            # print('ex FFD:',ex['FFD'])
            # print('new FFD:',new_ex['FFD'])
            name_dict['FFDs'].append(new_ex['FFD'])
            name_dict['GDs'].append(new_ex['GD'])
            name_dict['GPTs'].append(new_ex['GPT'])
            name_dict['TRTs'].append(new_ex['TRT'])
            name_dict['nFixs'].append(new_ex['nFix'])
        l = len(name_dict['words'])
        print('length:',l)
        for i in range(l):
            temp_input=InputExample(
                        words=name_dict['words'][i],
                        heads=name_dict['heads'][i],
                        dep_tags=name_dict['dep_tags'][i],
                        pos_tags=name_dict['pos_tags'][i],
                        FFDs=name_dict['FFDs'][i],
                        GDs=name_dict['GDs'][i],
                        GPTs=name_dict['GPTs'][i],
                        TRTs=name_dict['TRTs'][i],
                        nFixs=name_dict['nFixs'][i],
                    )
            examples.append(temp_input)
    print(num)
    ffd_length = 0
    gd_length = 0
    gpt_length = 0
    trt_length = 0
    nfix_length = 0

    for ffds in name_dict['FFDs']:
        for ffd in ffds:
            if ffd > ffd_length:
                ffd_length = ffd
    for gds in name_dict['GDs']:
        for gd in gds:
            if gd > gd_length:
                gd_length = gd
    for gpts in name_dict['GPTs']:
        for gpt in gpts:
            if gpt > gpt_length:
                gpt_length = gpt
    for trts in name_dict['TRTs']:
        for trt in trts:
            if trt > trt_length:
                trt_length = trt
    for nfixs in name_dict['nFixs']:
        for nfix in nfixs:
            if nfix > nfix_length:
                nfix_length = nfix
    all_length = max(ffd_length, gd_length, gpt_length, trt_length, nfix_length)
    # print('======train_examples=======')
    # for each in train_examples:
    #     print(each.words)
    # print('======val_examples=======')
    # for each in val_examples:
    #     print(each.words)
    eye_length = {}
    eye_length['ffd'] = ffd_length
    eye_length['gd'] = gd_length
    eye_length['gpt'] = gpt_length
    eye_length['trt'] = trt_length
    eye_length['nfix'] = nfix_length
    eye_length['all'] = all_length
    return examples, eye_length  # examples

def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_segment_id=0,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        use_syntax=False,
        revise_edge=False,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # print(label_list)
    # label_map = {label: i for i, label in enumerate(label_list)}
    special_tokens_count = 3 if sep_token_extra else 2
    # print('label_map:',label_map)
    features = []
    over_length_examples = 0
    wrong_examples = 0
    print('examples:',len(examples))
    for (ex_index, example) in enumerate(examples):
        print(ex_index/len(examples))
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        if 0 not in example.heads:
            wrong_examples += 1
            continue

        tokens, heads,  dep_tags, pos_tags,\
        FFDs, GDs, GPTs, TRTs, nFixs = process_sentence(
            example.words,
            example.heads,
            example.dep_tags,
            example.pos_tags,
            tokenizer,
            example.FFDs,
            example.GDs,
            example.GPTs,
            example.TRTs,
            example.nFixs
        )
        # print('here:',len(label_ids),len(FFDs))
        # print('pos_tags:',pos_tags)
        orig_text_len = len(tokens)
        root_idx = heads.index(0)
        text_offset = 1  # text_a follows <s>
        # So, we add 1 to head indices
        heads = np.add(heads, text_offset).tolist()
        # HEAD(<text_a> root) = index of <s> (1-based)
        heads[root_idx] = 1

        if len(tokens) > max_seq_length - special_tokens_count:
            # assert False  # we already truncated sequence
            # print("truncate token", len(tokens), max_seq_length, special_tokens_count)
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            dep_tags = dep_tags[: (max_seq_length - special_tokens_count)]
            pos_tags = pos_tags[: (max_seq_length - special_tokens_count)]
            FFDs = FFDs[: (max_seq_length - special_tokens_count)]
            GDs = GDs[: (max_seq_length - special_tokens_count)]
            GPTs = GPTs[: (max_seq_length - special_tokens_count)]
            TRTs = TRTs[: (max_seq_length - special_tokens_count)]
            nFixs = nFixs[: (max_seq_length - special_tokens_count)]
            heads = heads[: (max_seq_length - special_tokens_count)]
            # label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            over_length_examples += 1
            # continue

        tokens += [tokenizer.sep_token]
        dep_tags += [tokenizer.sep_token]
        pos_tags += [tokenizer.sep_token]
        # label_ids += [pad_token_label_id]
        FFDs += [0]
        GDs += [0]
        GPTs += [0]
        TRTs += [0]
        nFixs += [0]
        # print('here 2:', len(label_ids), len(FFDs))
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [tokenizer.sep_token]
            dep_tags += [tokenizer.sep_token]
            pos_tags += [tokenizer.sep_token]
            # label_ids += [pad_token_label_id]
            FFDs += [0]
            GDs += [0]
            GPTs += [0]
            TRTs += [0]
            nFixs += [0]
        # print('--tokens 1--')
        # print(tokens)
        segment_ids = [sequence_a_segment_id] * len(tokens)
        # print('here 3:', len(label_ids), len(FFDs))
        # cls_token_at_begining
        tokens = [tokenizer.cls_token] + tokens
        dep_tags = [tokenizer.cls_token] + dep_tags
        pos_tags = [tokenizer.cls_token] + pos_tags
        # label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids
        FFDs = [0] + FFDs
        GDs = [0] + GDs
        GPTs = [0] + GPTs
        TRTs = [0] + TRTs
        nFixs = [0] + nFixs
        # print('here 4:', len(label_ids), len(FFDs))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print('--tokens 2--')
        # print(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # label_ids = ([pad_token_label_id] * padding_length) + label_ids
            FFDs = ([0]* padding_length) + FFDs
            GDs = ([0]* padding_length) + GDs
            GPTs = ([0]* padding_length) + GPTs
            TRTs = ([0]* padding_length) + TRTs
            nFixs = ([0]* padding_length) + nFixs
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            # label_ids += [pad_token_label_id] * padding_length
            FFDs += [0] * padding_length
            GDs += [0] * padding_length
            GPTs += [0] * padding_length
            TRTs += [0] * padding_length
            nFixs += [0] * padding_length
        # print('here 5:', len(label_ids), len(FFDs))
        # if example.langs and len(example.langs) > 0:
        #     langs = [example.langs[0]] * max_seq_length
        # else:
        #     # print("example.langs", example.langs, example.words, len(example.langs))
        #     # print("ex_index", ex_index, len(examples))
        #     langs = None

        # print('length:',len(FFDs),len(GDs),len(GPTs),len(TRTs),len(nFixs))
        # print('max_seq_length',max_seq_length)
        # print('input_ids:',input_ids)
        # print('input_mask:',input_mask)
        # print('label_ids:',label_ids)
        # print('++++++')
        # print('FFD:',FFDs)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(label_ids) == max_seq_length
        # assert len(langs) == max_seq_length
        assert len(FFDs) == max_seq_length
        assert len(GDs) == max_seq_length
        assert len(GPTs) == max_seq_length
        assert len(TRTs) == max_seq_length
        assert len(nFixs) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        #     logger.info("langs: {}".format(langs))

        one_ex_features = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            # label_ids=label_ids,
        )
        one_ex_features.FFDs=FFDs
        one_ex_features.GDs = GDs
        one_ex_features.GPTs = GPTs
        one_ex_features.TRTs = TRTs
        one_ex_features.nFixs = nFixs
        # print('label_ids:',label_ids)
        if use_syntax:
            #####################################################
            # prepare the UPOS and DEPENDENCY tag tensors
            #####################################################
            dep_tag_ids = deptag_to_id(dep_tags, tokenizer=str(type(tokenizer)))
            pos_tag_ids = upos_to_id(pos_tags, tokenizer=str(type(tokenizer)))
            # print('util pos_tag_ids:',pos_tag_ids)

            if pad_on_left:
                dep_tag_ids = ([0] * padding_length) + dep_tag_ids
                pos_tag_ids = ([0] * padding_length) + pos_tag_ids
            else:
                dep_tag_ids += [0] * padding_length
                pos_tag_ids += [0] * padding_length

            assert len(input_ids) == len(dep_tag_ids)
            assert len(input_ids) == len(pos_tag_ids)
            assert len(dep_tag_ids) == max_seq_length
            assert len(pos_tag_ids) == max_seq_length

            # print('---pos_tag_ids---:',pos_tag_ids)
            one_ex_features.pos_tag_ids = pos_tag_ids
            one_ex_features.dep_tag_ids = dep_tag_ids

            #####################################################
            # form the tree structure using head information
            #####################################################
            heads = [0] + heads + [1, 1] if sep_token_extra else [0] + heads + [1]
            # print(len(heads),heads)
            # print(len(tokens),tokens)
            assert len(tokens) == len(heads)
            root, nodes = head_to_tree(heads, tokens)
            assert len(heads) == root.size()
            sep_token_indices = [i for i, x in enumerate(tokens) if x == tokenizer.sep_token]
            depths = [nodes[i].depth() for i in range(len(nodes))]
            depths = np.asarray(depths, dtype=np.int32)

            one_ex_features.root = root
            one_ex_features.depths = depths
            one_ex_features.sep_token_indices = sep_token_indices

        features.append(one_ex_features)
    print('len(features):',len(features))
    if over_length_examples > 0:
        logger.info('{} examples are discarded due to exceeding maximum length'.format(over_length_examples))
    if wrong_examples > 0:
        logger.info('{} wrong examples are discarded'.format(wrong_examples))
    return features

def multivar_continue_KL_divergence(p, q):
    a = np.log(np.linalg.det(q[1])/np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)

def euclidean(p,q):
    '''
    :param p: list
    :param q: list
    :return:
    '''
    #如果特征长度不同，不计算相似度
    if(len(p) != len(q)):
        raise Exception("feature length must be the same")

    p,q = np.array(p),np.array(q)
    d1 = np.sqrt(np.sum(np.square(p - q)))

    return d1

def exponent_distance(p,q,y):
    '''
    :param p: list
    :param q: list
    :param y: 缩放因子
    :return:
    '''
    p, q = np.array(p), np.array(q)
    d = euclidean(p,q)
    d1 = np.exp(-y * d)
    return d1


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)

def Wasserstein(p,q):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(p,q)

class SequenceDataset(Dataset):
    def __init__(self, features,revise_edge):
        self.features = features
        self.revise_edge = revise_edge

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """Generates one sample of data"""
        feature = self.features[index]
        # print('feature:',feature)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        # labels = torch.tensor(feature.label_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.segment_ids, dtype=torch.long)

        FFDs = torch.tensor(feature.FFDs, dtype=torch.long)
        GDs = torch.tensor(feature.GDs, dtype=torch.long)
        GPTs = torch.tensor(feature.GPTs, dtype=torch.long)
        TRTs = torch.tensor(feature.TRTs, dtype=torch.long)
        nFixs = torch.tensor(feature.nFixs, dtype=torch.long)

        dist_matrix = None
        depths = None
        dep_tag_ids = None
        pos_tag_ids = None

        if feature.root is not None:
            # print('==============')
            # print('feature:',feature)
            # print('feature.dep_tag_ids:',feature.dep_tag_ids)
            # print('feature.pos_tag_ids:',feature.pos_tag_ids)
            dep_tag_ids = torch.tensor(feature.dep_tag_ids, dtype=torch.long)
            pos_tag_ids = torch.tensor(feature.pos_tag_ids, dtype=torch.long)
            dist_matrix = root_to_dist_mat(feature.root)
            if feature.trunc_token_ids is not None:
                dist_matrix = np.delete(dist_matrix, feature.trunc_token_ids, 0)  # delete rows
                dist_matrix = np.delete(dist_matrix, feature.trunc_token_ids, 1)  # delete columns

            dist_matrix = torch.tensor(dist_matrix, dtype=torch.long)  # seq_len x seq_len x max-path-len
            # print('---------before dist_matrix-------')
            # print(dist_matrix)

        # if self.revise_edge == 'eye':
        eye_dist_matrix = []
        scale = 1
        # print('feature.FFDs:',feature.FFDs)
        minn=9999
        maxx=-1
        if feature.FFDs is not None:
            # print('feature.FFDs:',feature.FFDs)
            ffd_leng = len(feature.FFDs)
            for j in range(ffd_leng):
                temp_list = []
                for k in range(ffd_leng):
                    p = [feature.FFDs[j], feature.GDs[j], feature.GPTs[j], feature.TRTs[j], feature.nFixs[j]]
                    q = [feature.FFDs[k], feature.GDs[k], feature.GPTs[k], feature.TRTs[k], feature.nFixs[k]]
                    # print(p)
                    # print(q)
                    # p = np.asarray(p)
                    # q = np.asarray(q)
                    # print(JS_divergence(p, q))
                    # sci = scipy.stats.entropy(p, q)
                    # sci = JS_divergence(p, q)
                    sci = Wasserstein(p,q)
                    # 先不归一化试一下
                    '''
                    if math.isnan(sci):
                        sci = 0
                    elif sci == 0:
                        sci = 1
                    else:
                        sci = 1 / sci
                    '''
                    # print('scipy:',sci)
                    # print(sci)
                    if sci >= maxx:
                        maxx = sci
                    if sci <= minn:
                        minn = sci
                    temp_list.append(sci)
                    # print('temp:',temp_list)
                eye_dist_matrix.append(temp_list)

        # print('---------dist_matrix 1-------')
        # print(dist_matrix)
        for i in range(len(eye_dist_matrix)):
            for j in range(len(eye_dist_matrix[i])):
                # print('before:',eye_dist_matrix[i][j])
                eye_dist_matrix[i][j] = (eye_dist_matrix[i][j]-minn)/(maxx-minn)
                # print('after:',eye_dist_matrix[i][j])
        # final_dist_matrix = dist_matrix + eye_dist_matrix
        # final_dist_matrix = torch.tensor(final_dist_matrix, dtype=torch.long)
        eye_dist_matrix = torch.tensor(eye_dist_matrix, dtype=torch.long)
        # print(dist_matrix)
        # print('---------dist_matrix 2-------')
        if feature.depths is not None:
            depths = feature.depths
            if feature.trunc_token_ids is not None:
                depths = np.delete(depths, feature.trunc_token_ids, 0)
            depths = torch.tensor(depths, dtype=torch.long)  # seq_len

        # print('pos_tag_ids:',pos_tag_ids)
        # print('FFDs:',FFDs)
        return [
            input_ids,
            attention_mask,
            token_type_ids,
            # labels,
            dep_tag_ids,
            pos_tag_ids,
            dist_matrix,
            eye_dist_matrix,
            depths,
            FFDs,
            GDs,
            GPTs,
            TRTs,
            nFixs,
        ]


def batchify(batch):
    """Receives a batch of SequencePairDataset examples"""
    input_ids = torch.stack([data[0] for data in batch], dim=0)
    attention_mask = torch.stack([data[1] for data in batch], dim=0)
    token_type_ids = torch.stack([data[2] for data in batch], dim=0)
    FFDs = torch.stack([data[8] for data in batch], dim=0)
    GDs = torch.stack([data[9] for data in batch], dim=0)
    GPTs = torch.stack([data[10] for data in batch], dim=0)
    TRTs = torch.stack([data[11] for data in batch], dim=0)
    nFixs = torch.stack([data[12] for data in batch], dim=0)

    dist_matrix = None
    eye_dist_matrix = None
    depths = None
    dep_tag_ids = None
    pos_tag_ids = None

    if batch[0][3] is not None:
        dep_tag_ids = torch.stack([data[3] for data in batch], dim=0)

    if batch[0][4] is not None:
        pos_tag_ids = torch.stack([data[4] for data in batch], dim=0)

    if batch[0][5] is not None:
        dist_matrix = torch.full(
            (len(batch), input_ids.size(1), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen, slen = data[5].size()
            dist_matrix[i, :slen, :slen] = data[5]
    if batch[0][6] is not None:
        eye_dist_matrix = torch.full(
            (len(batch), input_ids.size(1), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen, slen = data[6].size()
            eye_dist_matrix[i, :slen, :slen] = data[6]

    if batch[0][7] is not None:
        depths = torch.full(
            (len(batch), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen = data[7].size(0)
            depths[i, :slen] = data[7]
    # print('dist_matrix.shape:',dist_matrix.shape)
    # print('eye_dist_matrix.shape:', eye_dist_matrix.shape)
    return [
        input_ids,
        attention_mask,
        token_type_ids,
        dep_tag_ids,
        pos_tag_ids,
        dist_matrix,
        eye_dist_matrix,
        depths,
        FFDs,
        GDs,
        GPTs,
        TRTs,
        nFixs
    ]
