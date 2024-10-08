import torch as th
import torch
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl, json
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
)
# import utils.batchify
from modeling_bert import BertForTokenClassification
# from processors.constants import *
from numpy import *

logger = logging.getLogger(__name__)
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

ALL_MODELS = tuple(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP)

MODEL_CLASSES = {
    "bert": (
        BertConfig, BertForTokenClassification, BertTokenizer
    ),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('-m', '--m', type=float, default=0.5, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=10)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='mr', choices=['R8_3','R8_1','R8_2','R8_5',
                                                        'R52_3', 'R52_1', 'R52_2','R52_5',
                                                        'ohsumed_2','ohsumed_1','ohsumed_3','ohsumed_5',
                                                        'mr_3','mr_1','mr_2','mr_4','mr_5','mr_6'])
# parser.add_argument('--checkpoint_dir', default='/home/leon/project_vincy/BertGCN-main/checkpoint/home/leon/project_vincy/BERT/bert_R52', help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--checkpoint_dir', default=None,
                    help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')

parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=128,
                    help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)
# ======================================================================================
parser.add_argument("--model_type", default='bert', type=str, required=False,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
# parser.add_argument("--init_checkpoint", default=None, type=str,
#                     help="initial checkpoint for train/predict")
parser.add_argument(
    "--use_dependency_tag",
    type=bool,
    default=False,
    help="Whether to use structural distance instead of linear distance",
)
parser.add_argument(
    "--use_pos_tag",
    type=bool,
    default=True,
    help="Whether to use structural distance instead of linear distance",
)
parser.add_argument(
    "--syntactic_layers",
    type=str, default='0,1,2',
    help="comma separated layer indices for syntax fusion",
)
parser.add_argument(
    "--num_syntactic_heads",
    default=2, type=int,
    help="Number of syntactic heads",
)
parser.add_argument(
    "--use_syntax",
    type=bool,
    default=True,
    help="Whether to use syntax-based modeling",
)
parser.add_argument(
    "--revise_edge",
    type=str,
    default='fuse',  # 'eye',
    help="Whether to revise edge of GAT modeling",
)
parser.add_argument(
    "--max_syntactic_distance",
    default=1, type=int,
    help="Max distance to consider during graph attention",
)
parser.add_argument(
    "--eye_max_syntactic_distance",
    default=0.5, type=float,
    help="Max distance to consider during graph attention",
)
parser.add_argument(
    "--num_gat_layer",
    default=4, type=int,
    help="Number of layers in Graph Attention Networks (GAT)",
)
parser.add_argument(
    "--num_gat_head",
    default=4, type=int,
    help="Number of attention heads in Graph Attention Networks (GAT)",
)
parser.add_argument(
    "--batch_normalize",
    action="store_true",
    help="Apply batch normalization to <s> representation",
)
parser.add_argument("--revise_gat", default='org', type=str, required=False,
                    help="eye feature of syntax")

# org:  O = Attention(Q , K , V , M, dk)
# eye:  O = Attention(Q , K + eye  , V + eye , M, dk)
# fuse: O = Attention(Q , K + 熵权法结果Wcog, V + 熵权法结果Wcog, M, dk)
parser.add_argument("--use_eye", default='org', type=str, required=False,
                    help="eye feature in attention")

# org:  O = Attention(Q , K , V , M, dk)
# single: QK用眼动特征，参考[Gated attention fusion network for multimodal sentiment classification]
# fuse: [text: Q , eye: KV] or [eye: Q , text: KV]
parser.add_argument("--use_fusion", default='org', type=str, required=False,
                    help="eye feature in attention")
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--model_name_or_path", default='/home/leon/project_vincy/BERT/bert', type=str,
                    required=False,
                    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                        ALL_MODELS))
parser.add_argument("--cache_dir", default=None, type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--train_data_file", default='new_mr_gaze.jsonl', type=str, required=False,
                    help="The input data dir. Should contain the training files for the NER/POS task.")
parser.add_argument("--max_seq_length", default=1024, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--do_lower_case", action="store_true",
                    help="Set this flag if you are using an uncased model.")
# parser.add_argument("--train_batch_size", default=1024, type=int,
#                     help="Batch size per GPU/CPU for training.")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
# ======================================================================================
args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:5')
device = gpu
logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    # num_labels=num_labels,
    cache_dir=args.cache_dir if args.cache_dir else None
)
tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None
)
pad_token_label_id = CrossEntropyLoss().ignore_index
# Data Preprocess
# adj_text,adj_gaze,adj_fuse, features, y_train, y_val, y_test,\
#            train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
adj_text, adj_gaze, features, y_train, y_val, y_test, \
train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
print('train_size, test_size:', train_size, test_size)
'''
all_example,all_eye_length = read_examples_from_file('data/corpus/new_'+dataset+'_all_gaze.jsonl')
train_example,train_eye_length = read_examples_from_file('data/corpus/new_'+dataset+'_train_gaze.jsonl')
test_example,test_eye_length = read_examples_from_file('data/corpus/new_'+dataset+'_test_gaze.jsonl')
all_features,train_features,test_features =[], [], []
features_lg = convert_examples_to_features(
            all_example,
            args.max_seq_length,
            tokenizer,
            cls_token_segment_id=0,
            sep_token_extra=bool(args.model_type in ["roberta", "xlm-roberta"]),
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            use_syntax=args.use_syntax,
            revise_edge=args.revise_edge
        )
all_features.extend(features_lg)
print('1:',len(all_features))
features_lg = convert_examples_to_features(
            train_example,
            args.max_seq_length,
            tokenizer,
            cls_token_segment_id=0,
            sep_token_extra=bool(args.model_type in ["roberta", "xlm-roberta"]),
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            use_syntax=args.use_syntax,
            revise_edge=args.revise_edge
        )
train_features.extend(features_lg)
features_lg = convert_examples_to_features(
            test_example,
            args.max_seq_length,
            tokenizer,
            cls_token_segment_id=0,
            sep_token_extra=bool(args.model_type in ["roberta", "xlm-roberta"]),
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            use_syntax=args.use_syntax,
            revise_edge=args.revise_edge
        )
test_features.append(features_lg)
all_dataset = SequenceDataset(all_features,args.revise_edge)
print('2:',len(all_dataset))
'''
print('==========all_dataset=============')
# gaze_dataset=[]
# for i in range(len(all_dataset)):
#     print(i/len(all_dataset))
#     gaze_dataset.append(all_dataset[i])
# gaze_dataset=th.Tensor(gaze_dataset)
# print(gaze_dataset.shape,gaze_dataset)
# print(all_dataset['input_ids'])
# train_dataset = SequenceDataset(train_features,args.revise_edge)
# test_dataset = SequenceDataset(test_features,args.revise_edge)
# print('gaze_train.shape:',gaze.shape)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]
''''''
# ====================================model==================================


import sys

sys.path.append('/home/leon/project_vincy/syn_bert/third_party')
from processors.constants import *

####################################
config.dep_tag_vocab_size = len(DEPTAG_SYMBOLS) + NUM_SPECIAL_TOKENS
config.pos_tag_vocab_size = len(POS_SYMBOLS) + NUM_SPECIAL_TOKENS
print('len--:', config.dep_tag_vocab_size, config.pos_tag_vocab_size)
config.use_dependency_tag = args.use_dependency_tag
config.use_pos_tag = args.use_pos_tag
# config.use_structural_loss = args.use_structural_loss
# config.struct_loss_coeff = args.struct_loss_coeff
config.num_syntactic_heads = args.num_syntactic_heads
config.syntactic_layers = args.syntactic_layers
config.max_syntactic_distance = args.max_syntactic_distance
config.eye_max_syntactic_distance = args.eye_max_syntactic_distance
config.use_syntax = args.use_syntax
config.revise_edge = args.revise_edge
config.batch_normalize = args.batch_normalize
config.num_gat_layer = args.num_gat_layer
config.num_gat_head = args.num_gat_head
config.revise_gat = args.revise_gat
config.eye_length = {}  # eye_length
config.use_eye = args.use_eye
config.device = device
####################################
'''
tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

if args.init_checkpoint:
    logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
    model = model_class.from_pretrained(args.init_checkpoint,
                                        config=config,
                                        cache_dir=args.init_checkpoint)
else:
    logger.info("loading from cached model = {}".format(args.model_name_or_path))
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

if args.use_syntax:
    if args.pretrained_gat:
        state_dict = torch.load(
            args.pretrained_gat, map_location=lambda storage, loc: storage
        )
        model.bert.encoder.gat_model.load_state_dict(state_dict)
        logger.info(" the gat model states are loaded from %s", args.pretrained_gat)

    if args.freeze_gat:
        for p in model.bert.encoder.gat_model.parameters():
            p.requires_grad = False
model.to(gpu)

# ====================================model==================================
'''
# instantiate model according to class number
if gcn_model == 'gcn':
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout, config=config)
else:
    model = BertGAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout)

if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

# load documents and compute input encodings
# corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
corpse_file = './data/'+ dataset +'/'+ dataset + '_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    #     print(input.keys())
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat(
    [attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])
print(input_ids.shape, input_ids)
print(attention_mask.shape, attention_mask)
# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

# build DGL Graph
adj = adj_text
# adj = adj_gaze
# adj = adj_fuse
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)

adj_norm = normalize_adj(adj_gaze + sp.eye(adj_gaze.shape[0]))
g_gaze = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g_gaze.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g_gaze.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)

# print(all_dataset[:, [0]])
# 把dataset融入进去
'''
print('all_dataset:',len(all_dataset))
all_dataloader = DataLoader(
        all_dataset,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=batchify,
    )
'''
# g.ndata['all_dataset']=th.LongTensor(all_dataset)
# print(g.ndata['all_dataset'].shape)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

g_gaze.ndata['label_train'] = th.LongTensor(y_train)
g_gaze.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))
# ------------------------------------------

# inputs["token_type_ids"] = batch[2] if args.model_type in ["bert"] else None
# inputs["labels"] = batch[3]
#
# if args.use_syntax:
#     inputs["dep_tag_ids"] = batch[4]
#     inputs["pos_tag_ids"] = batch[5]
#     inputs["dist_mat"] = batch[6]
#     inputs["eye_dist_mat"] = batch[7]
#     inputs["tree_depths"] = batch[8]
#     # print('dist_matrix.shape 2:', inputs["dist_mat"].shape)
#     # print('eye_dist_matrix.shape 2:', inputs["eye_dist_mat"].shape)
#
# inputs['FFDs'] = batch[9]
# inputs['GDs'] = batch[10]
# inputs['GPTs'] = batch[11]
# inputs['TRTs'] = batch[12]
# inputs['nFixs'] = batch[13]

# ------------------------------------------
logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    # targets = targets.cuda().data.cpu().numpy()
    # outputs = outputs.cuda().data.cpu().numpy()
    targets = np.array(targets)
    outputs = np.array(outputs)

    # print(targets)
    # print(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne.png'), bbox_inches='tight')
    print('done!')


# Training
def update_feature():
    # print('=================update_feature=================')
    global model, g, g_gaze, doc_mask, all_dataset
    # print(all_dataset[0].token_type_ids)
    # print(all_dataset['token_type_ids'][doc_mask])
    print('----------------------------------')
    print(g.ndata['input_ids'][doc_mask].shape)
    # print(g.ndata['input_ids'][doc_mask])
    # print('=========================')
    # print(g.ndata['input_ids'])
    # print('==========================')
    # temp_dep_tag_ids,temp_pos_tag_ids,temp_token_type_ids,temp_dist_mat,\
    # temp_eye_dist_mat,temp_tree_depths,temp_FFDs,temp_GDs,temp_GPTs,\
    # temp_TRTs,temp_nFixs=[],[],[],[],[],[],[],[],[],[],[]
    # print(all_dataset[0][0])
    # print(len(all_dataset),len(all_dataset[0]))
    '''
    input_ids,attention_mask,token_type_ids,dep_tag_ids,
    pos_tag_ids,dist_matrix,eye_dist_matrix,depths,
    FFDs,GDs,GPTs,TRTs,nFixs,'''
    # print(np.array(all_dataset))

    # all_dataset_mat=mat(all_dataset)
    # temp_dep_tag_ids=all_dataset_mat[:,1].tolist()
    # print(temp_dep_tag_ids)
    # print('++++++++++++++++++++++++++++++++')
    # for i in range(len(all_dataset)):
    #     print(i/len(all_dataset))
    #     temp_dep_tag_ids.append(all_dataset[i][3])
    #     temp_pos_tag_ids.append(all_dataset[i][4])
    #     temp_token_type_ids.append(all_dataset[i][2])
    #     temp_dist_mat.append(all_dataset[i][5])
    #     temp_eye_dist_mat.append(all_dataset[i][6])
    #     temp_tree_depths.append(all_dataset[i][7])
    #     temp_FFDs.append(all_dataset[i][8])
    #     temp_GDs.append(all_dataset[i][9])
    #     temp_GPTs.append(all_dataset[i][10])
    #     temp_TRTs.append(all_dataset[i][11])
    #     temp_nFixs.append(all_dataset[i][12])
    #
    # temp_dep_tag_ids=th.Tensor(temp_dep_tag_ids)
    # temp_pos_tag_ids=th.Tensor(temp_pos_tag_ids)
    # temp_token_type_ids=th.Tensor(temp_token_type_ids)
    # temp_dist_mat=th.Tensor(temp_dist_mat)
    # temp_eye_dist_mat=th.Tensor(temp_eye_dist_mat)
    # temp_tree_depths=th.Tensor(temp_tree_depths)
    # temp_FFDs=th.Tensor(temp_FFDs)
    # temp_GDs=th.Tensor(temp_GDs)
    # temp_GPTs=th.Tensor(temp_GPTs)
    # temp_TRTs=th.Tensor(temp_TRTs)
    # temp_nFixs=th.Tensor(temp_nFixs)

    # print('[doc_mask]:',temp_token_type_ids[doc_mask])
    # no gradient needed, uses a large batchsize to speed up the process
    ''''''
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask],
                           g.ndata['attention_mask'][doc_mask],
                           # all_dataset,
                           # temp_token_type_ids,
                           # temp_dep_tag_ids,
                           # temp_pos_tag_ids,
                           # temp_dist_mat,
                           # temp_eye_dist_mat,
                           # temp_tree_depths,
                           # temp_FFDs,
                           # temp_GDs,
                           # temp_GPTs,
                           # temp_TRTs,
                           # temp_nFixs,
                           ),
        batch_size=args.batch_size,
    )
    # train_iterator = trange(int(args.nb_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # # set_seed(args)  # Add here for reproductibility (even between python 2 and 3)
    #
    # for _ in train_iterator:
    #     epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        # epoch_iterator = tqdm(all_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for i, batch in enumerate(dataloader):
            # print('i,',i)
            # batch = tuple(t.to(device) if t is not None else None for t in batch)
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            # input_ids, attention_mask = [x.to(gpu) for x in batch]
            '''
            batch = tuple(t.to(device) if t is not None else None for t in batch)
            print(batch)
            inputs = dict()
            inputs["input_ids"]=batch[0]
            inputs["attention_mask"]=batch[1]
            inputs["token_type_ids"] = batch[2] if args.model_type in ["bert"] else None
            # inputs["labels"] = batch[3]

            if args.use_syntax:
                inputs["dep_tag_ids"] = batch[3]
                inputs["pos_tag_ids"] = batch[4]
                inputs["dist_mat"] = batch[5]
                inputs["eye_dist_mat"] = batch[6]
                # inputs["tree_depths"] = batch[7]
                # print('dist_matrix.shape 2:', inputs["dist_mat"].shape)
                # print('eye_dist_matrix.shape 2:', inputs["eye_dist_mat"].shape)

            inputs['FFDs'] = batch[8]
            inputs['GDs'] = batch[9]
            inputs['GPTs'] = batch[10]
            inputs['TRTs'] = batch[11]
            inputs['nFixs'] = batch[12]

            inputs["token_type_ids"] = all_dataset[i][2] if args.model_type in ["bert"] else None

            if args.use_syntax:
                inputs["dep_tag_ids"] = all_dataset[i][3]
                inputs["pos_tag_ids"] = all_dataset[i][4]
                inputs["dist_mat"] = all_dataset[i][5]
                inputs["eye_dist_mat"] = all_dataset[i][6]
                inputs["tree_depths"] = all_dataset[i][7]
                # print('dist_matrix.shape 2:', inputs["dist_mat"].shape)
                # print('eye_dist_matrix.shape 2:', inputs["eye_dist_mat"].shape)

            inputs['FFDs'] = all_dataset[i][8]
            inputs['GDs'] = all_dataset[i][9]
            inputs['GPTs'] = all_dataset[i][10]
            inputs['TRTs'] = all_dataset[i][11]
            inputs['nFixs'] = all_dataset[i][12]
            # token_type_ids=all_dataset[i][2]
            # dep_tag_ids=all_dataset[i][3]
            # pos_tag_ids=all_dataset[i][4]
            # dist_mat=all_dataset[i][5]
            # eye_dist_mat=all_dataset[i][6]
            # tree_depths=all_dataset[i][7]
            # FFDs=all_dataset[i][8]
            # GDs=all_dataset[i][9]
            # GPTs=all_dataset[i][10]
            # TRTs=all_dataset[i][11]
            # nFixs=all_dataset[i][12]
            '''
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]

            # output = model.bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0][:, 0]
            # output = model.bert(**inputs)[0][:, 0,:]
            # output = model.bert(**inputs)[0][:, 0]
            # tmp_eval_loss, logits = outputs[:2]
            # cls_list.append(output.cpu())
            cls_list.append(output)
        cls_feat = th.cat(cls_list, axis=0).to(device)
    # g = g.to(cpu)
    g = g.to(device)
    g_gaze = g_gaze.to(device)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    g_gaze.ndata['cls_feats'][doc_mask] = cls_feat
    return g, g_gaze


optimizer = th.optim.Adam([
    {'params': model.bert_model.parameters(), 'lr': bert_lr},
    {'params': model.classifier.parameters(), 'lr': bert_lr},
    {'params': model.gcn.parameters(), 'lr': gcn_lr},
], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

y_pred_list = []#np.empty([1,1],dtype=int)
y_true_list = []#np.empty([1,1],dtype=int)
num = 1
def train_step(engine, batch):
    # print('=================train_step=================')
    global model, g, g_gaze, optimizer, all_dataset, train_dataset
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    g_gaze = g_gaze.to(gpu)
    optimizer.zero_grad()
    (idx,) = [x.to(gpu) for x in batch]
    # print('idx:',idx)
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    # print(idx)
    # ---------------------
    # inputs = dict()
    # inputs["token_type_ids"] = train_dataset[idx][2] if args.model_type in ["bert"] else None
    #
    # if args.use_syntax:
    #     inputs["dep_tag_ids"] = train_dataset[idx][3]
    #     inputs["pos_tag_ids"] = train_dataset[idx][4]
    #     inputs["dist_mat"] = train_dataset[idx][5]
    #     inputs["eye_dist_mat"] = train_dataset[idx][6]
    #     inputs["tree_depths"] = train_dataset[idx][7]
    #     # print('dist_matrix.shape 2:', inputs["dist_mat"].shape)
    #     # print('eye_dist_matrix.shape 2:', inputs["eye_dist_mat"].shape)
    #
    # inputs['FFDs'] = train_dataset[idx][8]
    # inputs['GDs'] = train_dataset[idx][9]
    # inputs['GPTs'] = train_dataset[idx][10]
    # inputs['TRTs'] = train_dataset[idx][11]
    # inputs['nFixs'] = train_dataset[idx][12]
    # ---------------------
    y_pred = model(g, g_gaze, idx, is_bert=False)[train_mask]
    # y_pred = model(**inputs)
    # print(y_pred)
    gcn_logit = y_pred
    y_pred = th.nn.Softmax(dim=1)(gcn_logit)
    y_pred = th.log(y_pred)

    y_true = g.ndata['label_train'][idx][train_mask]
    # tsne = TSNE(random_state=0)
    # gcn_logit = gcn_logit.cuda().data.cpu().numpy()
    # # out = tsne.fit_transform(gcn_logit)
    # # out = out.cuda().data.cpu().numpy()
    # # print(out)
    # y_true_1 = y_true.cuda().data.cpu().numpy()
    # global y_pred_list,y_true_list,num
    # for each in gcn_logit.tolist():
    #     y_pred_list.append(each)
    # for each in y_true_1.tolist():
    #     y_true_list.append(each)
    # print('======y_pred_list========')
    # print(y_pred_list)
    # print('=======y_true_list=======')
    # print(y_true_list)
    '''
    if num == 1:
        y_pred_list = gcn_logit
        y_true_list = y_true_1
        num = 0
    else:
        y_pred_list=np.concatenate(y_pred_list,gcn_logit)
        # print(y_pred_list)
        # y_pred_list.append(gcn_logit.tolist())
        print('++++++++++y true+++++++')
        y_true_list = np.concatenate(y_true_list,y_true_1)
        print(y_true_list)
    '''
    # tsne_plot('./', y_true, y_pred)
    # exit()
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    # print('train_loss, train_acc:',train_loss, train_acc)
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    # print('=================reset_graph=================')
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    # print('=================test_step=================')
    global model, g, g_gaze
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        g_gaze = g_gaze.to(gpu)
        (idx,) = [x.to(gpu) for x in batch]
        y_pred = model(g, g_gaze, idx)
        y_true = g.ndata['label'][idx]
        # print(y_pred)
        # print(y_true)
        # tsne_plot('./', y_true, y_pred)
        # exit()
        gcn_logit = y_pred
        y_pred = th.nn.Softmax(dim=1)(gcn_logit)
        y_pred = th.log(y_pred)

        tsne = TSNE(random_state=0)
        gcn_logit = gcn_logit.cuda().data.cpu().numpy()
        # out = tsne.fit_transform(gcn_logit)
        # out = out.cuda().data.cpu().numpy()
        # print(out)
        y_true_1 = y_true.cuda().data.cpu().numpy()
        global y_pred_list, y_true_list, num
        for each in gcn_logit.tolist():
            y_pred_list.append(each)
        for each in y_true_1.tolist():
            y_true_list.append(each)

        return y_pred, y_true


evaluator = Engine(test_step)
metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    # print('=================log_training_result=================')
    # tsne_plot('./', y_true_list, y_pred_list)
    # exit()
    # out = y_pred_list[0]
    # for i in range(len(y_pred_list)):
    #     if i!=0:
    #         out = np.concatenate(out,y_pred_list[i])
    # y_true_1 = y_true_list[0]
    # for i in range(len(y_true_list)):
    #     if i != 0:
    #         y_true_1 = np.concatenate(y_true_1, y_true_list[i])
    # out = np.array(y_pred_list)
    # y_true_1=np.array(y_true_list)


    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
            .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc

    global y_pred_list, y_true_list
    out = np.array(y_pred_list)
    length = len(out[0])
    y_true_1 = np.array(y_true_list)
    print(out)
    tsne = TSNE()
    out = tsne.fit_transform(out)
    print('=======')
    print(y_true_1)
    fig = plt.figure()
    print(length)

    R8_label_dict_5 = ['earn', 'interest', 'ship', 'trade', 'crude', 'acq', 'money-fx', 'grain']
    R8_label_dict_2 = ['acq','trade','grain','earn','interest','ship','crude','money-fx']

    for i in range(length):
        indices = y_true_1 == i
        # new_x = []
        # new_y = []
        print(indices)
        print(out[indices])
        # for j in range(len(indices)):
        #     if j==True:
        #         new_x.append(out[i][j])
        #     else:

        print(out[indices].T)
        x, y = out[indices].T
        plt.scatter(x, y,label=str(i))
    plt.legend()
    plt.savefig(str(trainer.state.epoch) + '-R8-TRT.png')

    y_pred_list, y_true_list = [], []


log_training_results.best_val_acc = 0
g, g_gaze = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)
