import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, adj_score_list, adj_score_sum, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.adj_score_list = adj_score_list
        self.adj_score_sum = adj_score_sum
        self.average = True
        
    def forward(self, nodes, to_neighs, num_sample=1000):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
#        if not num_sample is None:
#            _sample = random.sample
#            samp_neighs = [_set(_sample(to_neigh, 
#                            num_sample,
#                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
#        else:
#            samp_neighs = to_neighs
        samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}    #모든 이웃 노드를 numbering
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        if self.average == True:
            column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
            row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
            mask[row_indices, column_indices] = 1
            if self.cuda:
                mask = mask.cuda()
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh)
        else:
            if self.cuda:
                mask = mask.cuda()
            for i in range(len(nodes)):
                node = nodes[i]
                neighs = to_neighs[i]
                for neigh in list(neighs):
                    mask[i, unique_nodes[neigh]] = self.adj_score_list[node][neigh] / self.adj_score_sum[node]
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats
