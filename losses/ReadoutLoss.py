import torch
from torch.nn import functional as F
from .LossFunction import LossFunction


b_xent = torch.nn.BCEWithLogitsLoss()

class ReadoutLoss(LossFunction):
    def __init__(self, tau):
        self.tau = tau

    def compute(self, z1, z2, adj, sub_g1, nns=False, alpha=0.5, metric="cosine", return_costs=False, **kwargs):
        subz_s, sub_gene_s = LossFunction.subg_centor(z1, z2, sub_g1)
        device = z1.device
        num = torch.randint(0, len(sub_g1) - 1, [len(sub_g1), ])
        if num[0] == 0:
            num[0] = 1
        for i in range(1, len(num)):
            if num[i] == i:
                num[i] -= 1
        subg2_s_n = subz_s[num]  # disrupt
        sub_gene_s_n = sub_gene_s[num]

        input1 = F.normalize(torch.mean(torch.cat((subz_s, subz_s, subz_s), dim=0), dim=1), dim=-1)
        input2 = F.normalize(torch.mean(torch.cat((sub_gene_s, subg2_s_n, sub_gene_s_n), dim=0), dim=1), dim=-1)
        logits = torch.exp(torch.sum(input1 * input2, dim=-1) / self.tau)

        lbl_1 = torch.ones(len(sub_g1)).to(device)
        lbl_2 = torch.zeros(len(sub_g1) * 2).to(device)
        lbl = torch.cat((lbl_1, lbl_2), 0).to(device)

        loss = b_xent(logits, lbl)

        return loss