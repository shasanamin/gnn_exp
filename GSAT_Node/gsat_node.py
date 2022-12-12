import sys
sys.path.append('../src')

import scipy
import torch
import torch.nn as nn
from torch_sparse import transpose
from torch_geometric.utils import is_undirected
from utils import reorder_like
from get_model import MLP
import torch_geometric as ptgeom #this is for extracting the computation subgraph

class GSAT(nn.Module):

    def __init__(self, clf, extractor, criterion, optimizer, learn_edge_att=True, final_r=0.7, decay_interval=10, decay_r=0.1):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = next(self.parameters()).device

        self.learn_edge_att = learn_edge_att
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        tot_loss = 0
        loss_dict_list = list() 
        
        
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        xwhole = np.zeros(np.size(871,)) #has to be of the size of 871 times dimension of the embedding
        edgewhole = np.zeros(1942) #hardcoded for the tree dataset : Data(x=[871, 10], edge_index=[2, 1942], y=[871], train_mask=[871], val_mask=[871], test_mask=[871])
        clfwhole = np.zeros(871) #hardcoded for tree dataset 

        ##looping for each node in the graph
        for i in range(dataset.data.x.size()[0]):
            node_index, edge_index, temp, temp = ptgeom.utils.k_hop_subgraph(i, 3, graphs) 
            att_log_logits = self.extractor(emb[node_index], edge_index, data.batch) #'NoneType' object
            att = self.sampling(att_log_logits, training)
            
            if self.learn_edge_att:
                if is_undirected(data.edge_index[edge_index]):
                    trans_idx, trans_val = transpose(edge_index, att, None, None, coalesced=False)
                    trans_val_perm = reorder_like(trans_idx, edge_index, trans_val) #trans_ids will not need edge_index
                    edge_att = (att + trans_val_perm) / 2
                else:
                    edge_att = att
            else:
                edge_att = self.lift_node_att_to_edge_att(att, edge_index)
           
            clf_logits, x = self.clf(data.x[node_index], edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
            #x is the node_embeddings returned by the second GNN - we can just add them up like with weights 1
            #but have to match it with the correct node_indices
            
            nn = 0
            for nodes in node_index:
                xwhole[nodes,:] = xwhole[nodes,:] +  x[nn]
                clfwhole[nodes]= clf_logits[nn]
                nn+=1
            #need to code the following according to the structure of edge_index...following is not correct structure-wise
            for edge in edge_index:
                edgewhole[edge] = edgewhole[edge]+edge_att 
            
            loss, loss_dict = self.__loss__(att, clf_logits, data.y[node_index], epoch)
           
        #simply adding up the losses and edge_att obtained from each node's computation graph
        tot_loss = tot_loss+loss
        loss_dict_list.append(loss_dict)
         
        return edgewhole, xwhole, tot_loss, loss_dict_list  #edge_att, loss, loss_dict, clf_logits
        #we don't need the clf_logits to be returned but need to match


    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.5)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits

