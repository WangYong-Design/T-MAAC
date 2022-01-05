from pickle import NONE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def multi_head_attention(q, k, v, mask=None):
#     # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
#     # k,v shape = (B, n_heads, N, key_dim)
#     # mask.shape = (B, group, N)

#     B, n_heads, n, key_dim = q.shape

#     # score.shape = (B, n_heads, n, N)
#     score = th.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

#     if mask is not None:
#         score += mask[:, None, :, :].expand_as(score)

#     shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
#     attn = th.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
#     return attn.reshape(*shp)

# def make_heads(qkv, n_heads):
#     shp = (qkv.size(0), qkv.size(1), n_heads, -1)
#     return qkv.reshape(*shp).transpose(1, 2)

# class EncoderLayer(nn.Module):
#     def __init__(self, embedding_dim, n_heads=8):
#         super(EncoderLayer, self).__init__()

#         self.n_heads = n_heads

#         self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
#             nn.Linear(embedding_dim * 4, embedding_dim))
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)

#     def forward(self, x, mask=None):
#         q = make_heads(self.Wq(x), self.n_heads)
#         k = make_heads(self.Wk(x), self.n_heads)
#         v = make_heads(self.Wv(x), self.n_heads)
#         x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
#         x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
#         x = x + self.feed_forward(x)
#         x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
#         return x


# class TransformerAgent(nn.Module):
#     # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
#     def __init__(self, obs_num, obs_dim, args):
#         super(TransformerAgent, self).__init__()
#         self.hidden_dim = args.hid_size
#         # self.out_hidden_dim = args.out_hid_size
#         self.attend_heads = args.attend_heads
#         assert (self.hidden_dim % self.attend_heads) == 0
#         self.n_layers = args.n_layers
#         self.attend_heads = args.attend_heads
#         self.args = args
#         self.obs_num = obs_num
#         self.obs_dim = obs_dim
#         self.init_projection_layer = nn.Linear(obs_dim, self.hidden_dim)
#         self.final_output_layer = nn.Linear(self.hidden_dim, args.action_dim)
#         self.attn_layers = nn.ModuleList([
#             EncoderLayer(embedding_dim=self.hidden_dim, n_heads=self.attend_heads)
#             for _ in range(self.n_layers)
#         ])
#         if args.layernorm:
#             self.layernorm = nn.LayerNorm(self.hidden_dim)
#         if args.hid_activation == 'relu':
#             self.hid_activation = nn.ReLU()
#         elif args.hid_activation == 'tanh':
#             self.hid_activation = nn.Tanh()

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.init_projection_layer.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()
#         # return th.zeros(1, self.args.agent_num, self.hidden_dim)

#     def forward(self, obs, hidden_state, agent_index, mask=None):
#         # obs : (b*n, self.obs_num, self.obs_dim)
#         # hidden_state : (b, n, self.hidden_dim )
#         # agent_index : (b*n, 1)
#         # mask : (b*n, 1, self.obs_num + 1)
#         x = self.init_projection_layer(obs)
#         x = th.cat((x, hidden_state.view(-1, self.hidden_dim).unsqueeze(dim=1)),dim=1)
#         for layer in self.attn_layers:
#             x = layer(x, mask)
#         index = agent_index.unsqueeze(dim=-1).expand(-1, 1, self.hidden_dim)
#         output = x.gather(1, index).contiguous().squeeze(dim=1) # (b*n, h)
#         output = self.final_output_layer(output)
#         h = x[:,-1:,:].contiguous().squeeze(dim=1) # (b*n, h)
#         return output, None, h
#         # x = self.final_projection_layer(x)  # (b, self.obs_num, out_hid_size)
#         # x = x.mean(dim=1)
#         # global_info = x.mean(dim=1, keepdim=True)
#         # glimpse_q = self.Wq(global_info)
#         # score = th.matmul(glimpse_q, x.transpose(1,2)) / np.sqrt(self.hidden_dim)
#         # score = F.softmax(score+mask, dim=2)
#         # final_embedding = th.matmul(score, x)
#         # return final_embedding.squeeze(dim=1)

class TransformerAgent(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hid_size, args.hid_size)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
        self.rnn = nn.GRUCell(args.hid_size, args.hid_size)
        self.final_output_layer = nn.Linear(args.hid_size, args.action_dim)
        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()
        # return th.zeros(1, self.args.agent_num, self.hidden_dim).cuda()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h_in = hidden_state.reshape(-1, self.args.hid_size)
        h = self.rnn(x, h_in)
        a = self.final_output_layer(h)
        return a, None, h