from pickle import NONE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = th.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = th.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)

def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(EncoderLayer, self).__init__()

        self.n_heads = n_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.norm2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, mask=None):
        q = make_heads(self.Wq(x), self.n_heads)
        k = make_heads(self.Wk(x), self.n_heads)
        v = make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class TransformerCritic(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, obs_size, action_dim, input_shape, output_shape, args, predict_dim=None):
        super(TransformerCritic, self).__init__()
        self.hidden_dim = args.hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.obs_size, self.action_dim = obs_size, action_dim
        self.n_layers = args.n_layers
        self.nagents = args.agent_num
        self.continuous = args.continuous
        self.attend_heads = args.attend_heads
        self.args = args
        self.predict_dcim = predict_dim
        
        self.init_projection_layer = nn.Linear(obs_size, args.hid_size)
        self.attn_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=self.hidden_dim, n_heads=self.attend_heads)
            for _ in range(self.n_layers)
        ])
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
        self.fc1 =nn.Linear(input_shape, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 =nn.Linear(self.hidden_dim, output_shape)

        if predict_dim is not None:
            self.pred_fc1 = nn.ModuleList()
            self.pred_fc2 = nn.ModuleList()
            for i in range(self.nagents):
                self.pred_fc1.append(nn.Linear(self.hidden_dim + self.nagents*self.action_dim, self.hidden_dim))
                self.pred_fc2.append(nn.Linear(self.hidden_dim, predict_dim[i]))

        if args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()


    def encoder(self, obs):
        x = self.init_projection_layer(obs)
        for layer in self.attn_layers:
            x = layer(x)
        return x

    def predict_voltage(self, enc, act, agent_id):
        B = enc.shape[0]
        x = th.cat((enc,act.view(B,-1)),dim=-1)
        x = self.hid_activation(self.pred_fc1[agent_id](x))
        x = self.pred_fc2[agent_id](x)
        return x

    def forward(self, x):
        x = self.fc1(x)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h = self.hid_activation(self.fc2(x))
        v = self.fc3(h)
        return v, h