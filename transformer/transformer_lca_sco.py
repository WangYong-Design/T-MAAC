import gc
from pickle import NONE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations  


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, args,embed_dim, dropout=0.0, max_len=322):
        """
        Construct the PositionalEncoding layer.
        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        self.pe = th.zeros(max_len, embed_dim).to("cuda")

        col = th.pow(10000,-th.arange(0,embed_dim,2)/embed_dim).to("cuda")
        row = th.arange(max_len).unsqueeze(1).to("cuda")
        m = row * col
        self.pe[:,0::2] = th.sin(m)
        self.pe[:,1::2] = th.cos(m)


    def forward(self):
        """
        Element-wise add positional embeddings to the input sequence.
        Inputs:
        Returns:
         - output: the input  positional encodings, of shape (S, D)
        """
        S, D = self.args.bus_num,self.args.lca_hid_size
        # Create a placeholder, to be overwritten by your code below.
        output = th.empty((S, D))

        output[:,:] = self.pe[:S,]
        output = self.dropout(output)

        return output


def multi_head_attention(q, k, v, mask=None,):
    # q shape = (B, n_,n_,n_heads, N, key_dim)   : N=40
    # k,v shape = (B, n_,n_,n_heads, N, key_dim)
    # mask.shape = (B, n_,n_, N,N)

    B,n_,n_, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_,n_,n_heads, N, N)
    score = th.matmul(q, k.transpose(5, 4))

    if mask is not None:
        score += mask[None,:, :,None, :, :].expand_as(score)

    shp = [q.size(0), q.size(1),q.size(2), q.size(-2),q.size(3)* q.size(-1)]
    # attn = th.matmul(score, v).transpose(1, 2)
    attn = th.matmul(F.softmax(score, dim=5), v).transpose(4, 3)   # (B, n_,n_,N,n_heads, key_dim)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), qkv.size(2),qkv.size(3),n_heads, -1)
    return qkv.reshape(*shp).transpose(-2,-3)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(EncoderLayer, self).__init__()

        self.n_heads = n_heads

        self.scaling = (embedding_dim // self.n_heads) ** -0.5

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False) 
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        q = make_heads(self.Wq(x), self.n_heads) * self.scaling
        k = make_heads(self.Wk(x), self.n_heads)
        v = make_heads(self.Wv(x), self.n_heads)
        
        # x = x + F.dropout(self.multi_head_combine(multi_head_attention(q, k, v, mask,attentive_bias)),p=0.1,training=self.training)
        x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
    
        # x = x + F.dropout(self.feed_forward(x),p=0.1,training=self.training)
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class TransformerlcaScore(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, obs_dim,node_dim, args):
        super(TransformerlcaScore, self).__init__()
        self.hidden_dim = args.lca_hid_size
        # self.out_hidden_dim = args.out_hid_size
        self.attend_heads = args.lca_attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.n_layers = args.lca_n_layers
        self.args = args
        self.n_ = args.agent_num
        self.max_lca_len = np.max(args.lca_len)
        self.agent_lca_pad = args.agent_lca_pad

        self.proj_obs = nn.Linear(obs_dim, self.hidden_dim)
        self.proj_node = nn.Linear(node_dim+self.hidden_dim,self.hidden_dim)
        self.padzeros = (0,0,0,1)

        self.attn_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=self.hidden_dim,
                         n_heads=self.attend_heads)
            for _ in range(self.n_layers)
        ])
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )


    def forward(self, obs, state,mask=None):
        # obs : (b, n_, obs_dim)
        # state : (b,323,bus_dim)
        bs,_,_ = obs.shape
        # obs_ = obs.transpose(0,1)

        obs_ = self.proj_obs(obs).transpose(0,1)
        node_ = F.pad(self.proj_node(state),self.padzeros,"constant",0)  # (bs,num_bus+1,hidden_dim)
        
        # attentive_bias = th.zeros((bs,self.n_,self.n_)).to("cuda")
        nodes_input = node_[:,self.agent_lca_pad.reshape(-1)].reshape(bs,self.n_,self.n_,\
                                                                self.max_lca_len,self.hidden_dim)

        agents_permutation = th.tensor(list(permutations([*obs_.tolist(),*obs_.tolist()],2))).to("cuda")
        agents_permutation = agents_permutation.reshape(2*self.n_-1, 2*self.n_, 2, bs, self.hidden_dim)
        agents_input = agents_permutation[self.n_-1:2*self.n_, :self.n_,:,:,:].permute(3,0,1,2,4)
        x = th.cat((agents_input,nodes_input),dim = -2).contiguous()     # x(bs,n_,n_,2+max_lca_len,hidden_dim)
        
        del agents_permutation,agents_input,nodes_input
        gc.collect()

        for layer in self.attn_layers:
            x = layer(x,mask)
            
        aggre_h = th.cat((x[:,:,:,0,:],x[:,:,:,1,:]),dim=3)    # shape(bs,n_,n_,hidden_dim*2)
        attentive_bias = self.out_layer(aggre_h)      
        # attentive_score = th.zeros((bs,self.n_,self.n_)).to(self.device)
        # for i in range(self.n_):
        #     agent_i_obs = obs[:,i,:]
        #     for j in range(self.n_):
        #         agent_j_obs = obs[:,j,:].to(self.device)
        #         lca_nodes = global_state[:,self.agent_lca[i][j]].clone().to(self.device)
        #         agent_obs_in = th.stack((agent_i_obs,agent_j_obs),dim=1).to(self.device)
        #         attentive_score[:,i,j] = self.attentive_score_tran(agent_obs_in,lca_nodes)


        #     x = th.cat((obs_,node_),dim = -2)    # x : (b,2+lca_len,hidden_dim) 
        #     for layer in self.attn_layers:
        #         x = layer(x)
        
        # aggre_h = th.cat((x[:,0,:],x[:,1,:]),dim=1)
        # attentive_bias = self.out_layer(aggre_h)

        return attentive_bias.squeeze(3)
