import torch
import torch.nn as nn
import numpy as np
from utilities.util import select_action, cuda_wrapper, batchnorm
from models.model import Model
from learning_algorithms.ddpg import DDPG
from collections import namedtuple
from critics.mlp_critic import MLPCritic


class MATD3(Model):
    def __init__(self, args, target_net=None):
        super(MATD3, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()

    def construct_value_net(self):
        # input_shape = (self.obs_dim + self.act_dim) * self.n_
        input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1 + self.n_
        # input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1
        output_shape = 1
        # self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_*2) ] )
        self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)

        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)

        # add agent id
        agent_ids = torch.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1) # shape = (b, n, n)
        agent_ids = cuda_wrapper(agent_ids, self.cuda_)
        obs_reshape = torch.cat( (obs_reshape, agent_ids), dim=-1 ) # shape = (b, n, n*o+n)

        # obs_reshape = obs_reshape_id.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*o+n)
        obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*o+n)
        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
        for bm in act_repeat:
            for i, nm in enumerate(bm):
                for j, a in enumerate(nm):
                    if j != i:
                        a = a.detach()
        act_reshape = act_repeat.contiguous().view( -1, np.prod(act.size()[1:]) ) # shape = (b*n, n*a)
        inputs = torch.cat( (obs_reshape, act_reshape), dim=-1 )
        ones = cuda_wrapper( torch.ones( inputs.size()[:-1] + (1,), dtype=torch.float ), self.cuda_)
        zeros = cuda_wrapper( torch.zeros( inputs.size()[:-1] + (1,), dtype=torch.float ), self.cuda_)
        inputs1 = torch.cat( (inputs, zeros), dim=-1 )
        inputs2 = torch.cat( (inputs, ones), dim=-1 )
        agent_value = self.value_dicts[0]
        values1, _ = agent_value(inputs1, None)
        values2, _ = agent_value(inputs2, None)
        values1 = values1.contiguous().view(batch_size, self.n_, 1)
        values2 = values2.contiguous().view(batch_size, self.n_, 1)

        return torch.cat([values1, values2], dim=0)

    def get_actions(self, state, status, exploration, actions_avail, target=False):
        if self.args.continuous:
            means, log_stds, _ = self.policy(state) if not target else self.target_net.policy(state)
            means[actions_avail == 0] = 0.0
            log_stds[actions_avail == 0] = 0.0
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_, 'clip': True if target else False})
            restore_mask = 1. - cuda_wrapper((actions_avail == 0).float(), self.cuda_)
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, _ = self.policy(state) if not target else self.target_net.policy(state)
            logits[actions_avail == 0] = -9999999
            # this follows the original version of sac: sampling actions
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False)
        _, next_actions, _, _ = self.get_actions(next_state, status='train', exploration=True, actions_avail=actions_avail, target=self.args.target)
        compose_pol = self.value(state, actions_pol)
        values_pol = compose_pol[:batch_size, :]
        values_pol = values_pol.contiguous().view(-1, self.n_)
        compose = self.value(state, actions)
        values1, values2 = compose[:batch_size, :], compose[batch_size:, :]
        values1 = values1.contiguous().view(-1, self.n_)
        values2 = values2.contiguous().view(-1, self.n_)
        next_compose = self.target_net.value(next_state, next_actions.detach())
        next_values1, next_values2 = next_compose[:batch_size, :], next_compose[batch_size:, :]
        next_values1 = next_values1.contiguous().view(-1, self.n_)
        next_values2 = next_values2.contiguous().view(-1, self.n_)
        returns = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        assert values_pol.size() == next_values1.size() == next_values2.size()
        assert returns.size() == values1.size() == values2.size()
        # update twin values by the minimized target q
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else torch.minimum(next_values1[i].detach(), next_values2[i].detach())
            else:
                next_return = torch.minimum(next_values1[i].detach(), next_values2[i].detach())
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas1 = returns - values1
        deltas2 = returns - values2
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        policy_loss = - advantages
        # policy_loss = policy_loss.mean(dim=0)
        # value_loss = torch.cat((deltas1.pow(2).mean(dim=0), deltas2.pow(2).mean(dim=0)), dim=-1)
        policy_loss = policy_loss.mean()
        value_loss = 0.5 * ( deltas1.pow(2).mean() + deltas2.pow(2).mean() )
        return policy_loss, value_loss, action_out
