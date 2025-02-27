from .LossFunction import LossFunction
import torch
import torch.nn.functional as F
import numpy as np

b_xent = torch.nn.BCEWithLogitsLoss()


class FOSSILLoss(LossFunction):
    def __init__(self, tau):
        self.tau = tau

    def compute(self, z1, z2, adj, sub_g1, alpha=0.5, reg=0.02, metric="cosine", **kwargs):
        subz_s, sub_gene_s = LossFunction.subg_centor(z1, z2, sub_g1)
        device = z1.device
        num = torch.randint(0, len(sub_g1) - 1, [len(sub_g1), ])
        if num[0] == 0:
            num[0] = 1
        for i in range(1, len(num)):
            if num[i] == i:
                num[i] -= 1
        subg2_s_n = subz_s[num]  # disrupt
        num = torch.randint(0, len(sub_g1) - 1, [len(sub_g1), ])
        if num[0] == 0:
            num[0] = 1
        for i in range(1, len(num)):
            if num[i] == i:
                num[i] -= 1
        sub_gene_s_n = sub_gene_s[num]

        input1 = torch.cat((subz_s, subz_s, subz_s), dim=0)
        input2 = torch.cat((sub_gene_s, subg2_s_n, sub_gene_s_n), dim=0)

        # adj
        subg1_adj = self.sub_adj(adj, sub_g1)
        input_adj = torch.cat((subg1_adj, subg1_adj, subg1_adj), dim=0)

        lbl_1 = torch.ones(len(sub_g1)).to(device)
        lbl_2 = torch.zeros(len(sub_g1) * 2).to(device)
        lbl = torch.cat((lbl_1, lbl_2), 0).to(device)

        fgwd = self.fGWD(input1.transpose(2, 1), input2.transpose(2, 1),
                                        input_adj, self.tau, alpha=alpha, reg=reg, metric=metric, **kwargs)

        logits = torch.exp(-fgwd / 0.01)
        ot_loss = b_xent(torch.squeeze(logits), lbl)
        l1 = self.semi_loss(z1, z2, **kwargs)
        l2 = self.semi_loss(z2, z1, **kwargs)
        node_level_loss = torch.mean((l1 + l2) * 0.5)
        loss = ot_loss + node_level_loss
        
        return loss
        
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
        
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def fGWD(self, X, Y, input_adj, tau, alpha, reg=0.5, metric="cosine", max_iter=20, tol=1e-9):
        M, Cs, Ct, p, q = self.setup_costs(X, Y, tau, input_adj, metric=metric)
        bs, m, n = M.size()
        device = M.device
        one_m = torch.ones(bs, m, 1).float().to(device)
        one_n = torch.ones(bs, n, 1).float().to(device)

        constC = (torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_n, 1, 2))
               + torch.bmm(one_m, torch.bmm(torch.transpose(q, 1, 2), torch.transpose(Ct**2, 1, 2))))

        def df(T):
            A = -torch.bmm(torch.bmm(Cs, T), 2*torch.transpose(Ct, 1, 2))
            return alpha * M + 2 * (1-alpha) * A
        
        cpt = 0
        err = 1e15

        T = torch.bmm(p, q.transpose(1, 2))
        while (err > tol and cpt < max_iter):
            Tprev = T

            # rows update
            T = T * torch.exp(df(T) / (-reg))
            row_scaling = p / torch.sum(T, 2, keepdim=True)
            T = row_scaling * T

            # column update
            T = T * torch.exp(df(T) / (-reg))
            column_scaling = q / torch.sum(T, 1, keepdim=True)
            T = column_scaling * T

            if cpt % 10 == 0:
                err = torch.norm(T - Tprev)

            cpt += 1

        if torch.isnan(T).any():
            print("Solver failed to produce a transport plan. You might want to increase the regularization parameter.")

        LxT = self.tensor_product(constC, Cs, Ct, T)
        fgwd = alpha * torch.sum(M * T, dim=(1, 2)) + (1-alpha) * torch.sum(LxT * T, dim=(1, 2))

        return fgwd
    
    def tensor_product(self, constC, Cs, Ct, T):
        A = - torch.bmm(torch.bmm(Cs, T), 2*torch.transpose(Ct, 1, 2))
        tens = constC + A
        return tens
    
    def gwggrad(self, constC, Cs, Ct, T):
        return 2 * self.tensor_product(constC, Cs, Ct, T)

    def setup_costs(self, X, Y, tau, input_adj, metric="cosine"):
        m = X.size(2)
        n = Y.size(2)
        bs = X.size(0)
        device = X.device
        p = (torch.ones(bs, m, 1) / m).to(device)
        q = (torch.ones(bs, n, 1) / n).to(device)
        if metric == "cosine":
            # Computing internal costs
            Cs = self.cos_batch(X, X, tau).float()
            #assert Cs.shape == input_adj.shape
            Cs = Cs * torch.exp(- input_adj / tau)
            #Cs = torch.exp(-input_adj/tau).cuda()
            Ct = self.cos_batch(Y, Y, tau).float()
            # Computing external costs
            cos_distance = self.cost_matrix_batch(X, Y, tau)  # (bs, m, n)

            beta = 0.1
            min_score = cos_distance.view(bs, -1).min(dim=-1)[0]
            max_score = cos_distance.view(bs, -1).max(dim=-1)[0]
            threshold = min_score + beta * (max_score - min_score)
            M = torch.nn.functional.relu(cos_distance - threshold.view(bs, 1, 1))  # (bs, m, n)
        elif metric == "l2":
            x2 = torch.sum(X**2, dim=1, keepdim=True)
            x2_t = torch.transpose(x2, 1, 2)
            y2 = torch.sum(Y**2, dim=1, keepdim=True)
            y2_t = torch.transpose(y2, 1, 2)
            xdotx = torch.bmm(torch.transpose(X, 1, 2), X)
            xdoty = torch.bmm(torch.transpose(X, 1, 2), Y)
            ydoty = torch.bmm(torch.transpose(Y, 1, 2), Y)
            Cs = x2_t - 2*xdotx + x2
            Ct = y2_t - 2*ydoty + y2
            M = x2_t - 2*xdoty + y2
        else:
            raise NotImplemented(f'metric "{metric}" not implemented') 

        return M, Cs, Ct, p, q