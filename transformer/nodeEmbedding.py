import math

import torch
import torch.nn as nn 

def init_params(module,n_layers):
    if isinstance(module,nn.Linear):
        module.weight.data.normal_(mean=0.0,std=0.02/math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module,nn.Embedding):
        module.weight.data.normal_(mean=0.0,std=0.2)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

class NodeEmbeddingFeature(nn.Module):
    """
    Compute the node feature for node in graph
    """
    def __init__(self,args):
        super().__init__()
        
        self.args = args
        self.bus_num = args.bus_num
        self.n_ = args.agent_num
        self.embed_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.attend_heads = args.attend_heads

        self.node_embedding = nn.Embedding(self.bus_num + 1,self.embed_dim*self.attend_heads,padding_idx=322)
        self.proj_bias = nn.Linear(self.embed_dim,1,bias = False)
        # self.proj_bias = nn.Linear(self.embed_dim*self.attend_heads,self.attend_heads,bias = False)

        self.agent_lca = torch.tensor(args.agent_lca,dtype=torch.long).to("cuda")
        self.lca_len = torch.tensor(args.lca_len,dtype = torch.long).to("cuda")

        self.apply(lambda module : init_params(module,self.n_layers))
        # self.node_embedding.weight[322] = 0.0


    def forward(self):
        " Compute edge attention bias"
        
        edge_embed = self.node_embedding(self.agent_lca).view(self.n_,self.n_,\
        torch.max(self.lca_len),self.attend_heads,self.embed_dim)

        edge_embed = edge_embed.sum(dim=2)/self.lca_len[:,:,None,None]  # (n_,n_,atten_heads,embed_dim)
        
        
        attentive_bias = self.proj_bias(edge_embed)  
        # attentive_bias = self.proj_bias(edge_embed.reshape(self.n_,self.n_,-1))  

        # for i in range(self.agent_num):
        #     lca_i = self.agent_lca[i]
            
        #     for j in range(self.agent_num):
        #         idx = torch.tensor(lca_i[j]).to("cuda")
        #         embed = self.node_embedding(idx).reshape(-1,self.embed_dim,self.attend_heads).mean(dim = 0)
        #         bias = self.proj_bias(embed.transpose(0,1)).squeeze(1)
        #         attentive_bias[i,j,:] = bias
        
        return attentive_bias.squeeze(-1).permute(2,0,1)
        # return attentive_bias








