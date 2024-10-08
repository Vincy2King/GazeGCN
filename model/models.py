import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT
from .modeling_bert import BertModel
import sys
sys.path.append('/home/leon/project_vincy/BertGCN-main/model')

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5,config=None):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.bert = BertModel(config)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g,g_gaze, idx,is_bert=False):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]

            '''
            if is_bert==True:
                cls_feats = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                pos_tag_ids=pos_tag_ids,
                dep_tag_ids=dep_tag_ids,
                dist_mat=dist_mat,
                eye_dist_mat=eye_dist_mat,
                FFDs=FFDs,
                GDs=GDs,
                GPTs=GPTs,
                TRTs=TRTs,
                nFixs=nFixs,
            )
            '''
            g.ndata['cls_feats'][idx] = cls_feats
            # g_gaze.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]

        # print(cls_feats.shape)
        # print(self.bert_model(input_ids, attention_mask)[0].shape)

        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        # print('idx2:',idx)
        gcn_logit= self.gcn(g.ndata['cls_feats'], g,g_gaze, g.edata['edge_weight'],g_gaze.edata['edge_weight'],self.bert_model(input_ids, attention_mask)[0])[idx]
        # print(gcn_output)
        # print('----gcn_logit-------')
        # print(gcn_logit)
        # exit()
        # gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        # print('before:',type(gcn_pred),gcn_pred.shape)
        # gcn_gaze_logit = self.gcn(g.ndata['cls_feats'], g_gaze, g_gaze, g_gaze.edata['edge_weight'])[idx]

        # print('logit:',gcn_logit,type(gcn_logit),type(gcn_gaze_logit))
        # gcn_logit = gcn_logit+gcn_gaze_logit
        # print('logit:', gcn_logit)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        # gaze_pred = th.nn.Softmax(dim=1)(gcn_gaze_logit)
        n = 0.4
        m = 0.3
        k = 0.3
        # print(type(gcn_pred),gcn_pred.shape)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        # pred = (gcn_pred + 1e-10) * m + cls_pred * n + (gaze_pred + 1e-10) * k
        pred = gcn_pred
        pred = th.log(pred)
        #
        # a = [aa.tolist() for aa in gcn_logit]  # 列表中元素由tensor变成列表。
        # print('-------after------')
        # print(torch.tensor(a))

        return gcn_logit#[pred,gcn_logit]
    
class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
