from re import X
from tkinter import E
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SacledDotProductAttention(nn.Module):
    def __init__(self,temp,dropout = 0):
        super().__init__()
        self.temp = temp
        self.dropout = dropout
    
    def forward(self,q,k,v,edges_feats,mask = None):
        B, n_heads, n, key_dim = q.shape

        # scores.shape = (B, n_heads, n, N,key_dim)
        # edges.shape = (B,n_heads,n,N,key_dim)

        q = q.reshape((B,n_heads,n,1,key_dim))
        k = k.reshape((B,n_heads,1,n,key_dim))
        scores = q * k * edges_feats
 
        logits = scores.sum(dim = -1)/ np.sqrt(q.size(-1))  # (B,n_head,n,N)

        if mask is not None:
            logits += mask[:, None, :, :].expand_as(logits)

        shp = [q.size(0), q.size(2), q.size(1) * q.size(-1)]
        attn = F.softmax(logits, dim=3)
        attn = torch.matmul(attn, v).transpose(1, 2)

        return attn.reshape(*shp),scores.transpose(1, 3).reshape(B, n, n,n_heads*key_dim)


class transformerlayer(nn.Module):
    def __init__(self,args,embed_dim,head_num) -> None:
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.head_num = head_num

        assert embed_dim % head_num ==0

        self.Q = nn.Linear(self.embed_dim,self.embed_dim)
        self.K = nn.Linear(self.embed_dim,self.embed_dim)
        self.V = nn.Linear(self.embed_dim,self.embed_dim)

        self.attention = SacledDotProductAttention(temp = np.power(self.embed_dim // self.head_num,self.args.dropout))

        self.O_h = nn.Linear(self.embed_dim,self.embed_dim)
        self.O_e = nn.Linear(self.embed_dim,self.embed_dim)

        self.h_feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(),
            nn.Dropout(p=self.args.dropout,training = self.training),
            nn.Linear(self.embed_dim * 2, self.embed_dim))
        
        self.e_feed_forward =  nn.Sequential(nn.Linear(self.embed_dim,self.embed_dim),nn.ReLU(),
                                            nn.Dropout(p=self.args.dropout,training = self.training),
                                            nn.Linear(self.embed_dim,self.embed_dim))
        # self.e_feed_forward2 = nn.Linear(self.embed_dim,self.embed_dim)

        if self.args.norm_type == "layernorm":
            self.h_norm1 = nn.LayerNorm(self.embed_dim)
            self.h_norm2 = nn.LayerNorm(self.embed_dim)

            self.e_norm1 = nn.LayerNorm(self.embed_dim)
            self.e_norm2 = nn.LayerNorm(self.embed_dim)
        elif self.args.norm_type == "batchnorm":
            self.h_norm1 = nn.BatchNorm1d(self.embed_dim)
            self.h_norm2 = nn.BatchNorm1d(self.embed_dim)

            self.e_norm1 = nn.BatchNorm1d(self.embed_dim)
            self.e_norm2 = nn.BatchNorm1d(self.embed_dim)
        else:
            NotImplementedError()
    
    def forward(self,nodes_feats,edges_feats,mask=None):
        # nodes_feats:(b,n,e)
        # edges_feats:(b,n,n,e)

        x = nodes_feats
        b,n_,_,_ = edges_feats.shape
        Q = self.make_heads(self.Q(nodes_feats),self.head_num)
        K = self.make_heads(self.K(nodes_feats),self.head_num)
        V = self.make_heads(self.V(nodes_feats),self.head_num)
        edges_input = edges_feats.reshape(b,n_,n_,self.head_num,-1).permute(0,3,1,2,4)

        atten,edges_out = self.attention(Q,K,V,edges_input,mask)
        
        x = x + self.O_h(F.dropout(atten,self.args.dropout,training=self.training))
        e = edges_feats + self.O_e(F.dropout(edges_out,self.args.dropout,training=self.training))

        x = self.h_norm1(x.view(-1,x.size(-1))).view(*x.size())
        e = self.e_norm1(e.view(-1,e.size(-1))).view(*e.size())

        h_in2,e_in2 = x,e
        x = h_in2+self.h_feed_forward(x)
        e = e_in2+self.e_feed_forward(e)

        x = self.h_norm2(x.view(-1,x.size(-1))).view(*x.size())
        e = self.e_norm2(e.view(-1,x.size(-1))).view(*e.size())
        
        # x = x + self.O_h(atten)
        # x = self.h_norm1(x.view(-1,x.size(-1))).view(*x.size())
        # x = F.dropout(x,self.args.dropout,training = self.training)
        # x = x + self.h_feed_forward(x)
        # x = self.h_norm2(x.view(-1,x.size(-1))).view(*x.size())

        # e = edges_feats + self.O_e(edges_out)
        # e = self.e_norm1(e.view(-1,e.size(-1))).view(*e.size())
        # e = F.dropout(e, self.args.dropout, training=self.training)
        # e = e + self.e_feed_forward(e)
        # e = self.e_norm2(e.view(-1,x.size(-1))).view(*e.size())

        return x,e


    def make_heads(self,qkv, n_heads):
        shp = (qkv.size(0), qkv.size(1), n_heads, -1)
        return qkv.reshape(*shp).transpose(1, 2)
        

class graphtransformer(nn.Module):
    def __init__(self,in_dim,args) -> None:
        super().__init__()
        self.args = args
        self.in_dim = in_dim
        self.hidd_dim = self.args.critic_hid_size
        self.n_layers = self.args.critic_n_layers
        self.head_num = self.args.attend_heads
        
        self.h_proj_layer = nn.Linear(self.in_dim,self.hidd_dim)
        self.e_proj_layer = nn.Linear(self.in_dim - self.args.action_dim,self.hidd_dim)

        self.atten_layer = nn.ModuleList([transformerlayer(self.args,self.hidd_dim,self.head_num) 
                                        for _ in range(self.n_layers)]) 

        self.reward_head = nn.Sequential(
            nn.Linear(self.hidd_dim, self.hidd_dim),
            nn.ReLU(),
            nn.Linear(self.hidd_dim, 1)
        )
        self.cost_head = nn.Sequential(
            nn.Linear(self.hidd_dim, self.hidd_dim),
            nn.ReLU(),
            nn.Linear(self.hidd_dim, 1)
        )
        # if args.layernorm:
        #     self.layernorm = nn.LayerNorm(self.hidden_dim)
        # if args.hid_activation == 'relu':
        #     self.hid_activation = nn.ReLU()
        # elif args.hid_activation == 'tanh':
        #     self.hid_activation = nn.Tanh()
    def forward(self,nodes_feats,edges_feats,mask=None):
        x = self.h_proj_layer(nodes_feats)
        e = self.e_proj_layer(edges_feats)
        for layer in self.atten_layer:
            x,e = layer(x,e,mask)
        
        pred_r = self.reward_head(x)  # (B,n,1)
        pred_c = self.cost_head(x)    # (B,N,1)

        return pred_r,pred_c





