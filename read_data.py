import torch



bias1 = torch.load("bias_test_1_1")
max_minus_min1 = torch.load("max_minus_min_test_1_1.pt")
score_ori1 = torch.load("score_ori_test_1_1.pt")
score_bias1 = torch.load("score_bias_test_1_1.pt")
max_bias_index1 = torch.load("max_bias_index_test_1_1.pt")
max_bias1 = torch.load("max_bias_test_1_1.pt")

bias2 = torch.load("bias_test_2.pt")
max_minus_min2 = torch.load("max_minus_min_test_2.pt")
score_ori2 = torch.load("score_ori_test_2.pt")
score_bias2 = torch.load("score_bias_test_2.pt")
max_bias_index2 = torch.load("max_bias_index_test_2.pt")
max_bias2 = torch.load("max_bias_test_2.pt")


agent_lca_pad = torch.load("agent_lca_pad.pt")
agent_lca = torch.load("agent_lca.pt")
print(True)


