import torch
import warnings
from typing import List, Union


class LossFunction:

    def compute(self, z1, z2, adj, sub_g1, loss_fn='default', nns=False, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def sub_adj(adj, sub_g1):
        """Get the adjacency matrix of all the sampled subgraphs in `sub_g1`
           Each row of `sub_g1` is a list of the nodes of a subgraph.

        Parameters
        ----------
        adj : Union[torch.Tensor, torch.SparseTensor]
            The adjacency matrix of the whole graph
        sub_g1 : List[List[int]]
            The list of the lists of nodes of each sampled subgraph
            Shape (bs, k)
            Where:
            bs is the number of generated subgraphs
            k is the number of nodes in each generated subgraph

        Returns
        -------
        torch.Tensor
            The adjacency lists of each sampled subgraph
            Shape (bs, k, k)
        """
        assert isinstance(adj, torch.Tensor)
        device = adj.device
        subg1_adj = torch.zeros(len(sub_g1), len(sub_g1[0]), len(sub_g1[0])).to(device)
        if adj.layout == torch.strided: # torch dense tensor
            for i in range(len(sub_g1)):
                subg1_adj[i] = adj[sub_g1[i]].t()[sub_g1[i]]
        
        elif adj.layout in [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc]: # torch sparse tensor
            for i in range(len(sub_g1)):
                indices = torch.LongTensor(sub_g1[i]).to(device)
                subg1_adj[i] = torch.index_select(torch.index_select(adj, 0, indices), 1, indices).to_dense()
        else:
            raise NotImplemented(f'Unknown storage format: {adj.layout}')
        
        return subg1_adj

    @staticmethod
    def subg_centor(z, z_g, sub_g1):
        """Get the embedding vectors and the genereted subgraph embedding vectors
           of the nodes of the sampled subgraphs in `sub_g1`

        Parameters
        ----------
        z : torch.Tensor
            The embeddings of the nodes
            Shape (bs, m, d)
            Where:
            bs is the batch size
            m is the number of nodes
            d is the embedding dimension
        z_g : torch.Tensor
            The embeddings of the subgraph of the nodes
            Shape (bs, m, d)
        sub_g1 : List[List[int]]
            The list of the lists of nodes of each sampled subgraph
            Shape (bs, k)
            Where:
            bs is the number of generated subgraphs
            k is the number of nodes in each generated subgraph

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
        """
        sub = [element for lis in sub_g1 for element in lis]
        subz = z[sub]
        subg = z_g[sub]

        sub_s = subz.reshape(len(sub_g1), len(sub_g1[0]), -1)
        subg_s = subg.reshape(len(sub_g1), len(sub_g1[0]), -1)
        return sub_s, subg_s

    @staticmethod
    def cost(L:torch.Tensor, T:torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.sum(L * T, -1), -1).unsqueeze(1)

    @staticmethod
    def tensor_product(constC, C1, C2, T):
        """_summary_

        Parameters
        ----------
        constC : torch.Tensor
            Shape (bs, m, n)
        C1 : torch.Tensor
            The internal costs matrix between the nodes of the first graph
            Shape (bs, m, m)
        C2 : torch.Tensor
            The internal costs matrix between the nodes of the first graph
            Shape (bs, n, n)
        T : torch.Tensor
            The transport plan
            Shape (bs, m, n)

        Returns
        -------
        torch.Tensor
            Shape (bs, m, n)
        """
        return constC - torch.bmm(torch.bmm(C1, T), torch.transpose(2 * C2, 1, 2))

    @staticmethod
    def tensor_product2(C1, C2, T):
        """_summary_

        Parameters
        ----------
        C1 : torch.Tensor
            The internal costs matrix between the nodes of the first graph
            Shape (bs, m, m)
        C2 : torch.Tensor
            The internal costs matrix between the nodes of the first graph
            Shape (bs, n, n)
        T : torch.Tensor
            The transport plan
            Shape (bs, m, n)

        Returns
        -------
        torch.Tensor
            Shape (bs, m, n)
        """
        bs, m, n = T.shape
        device = C1.device
        one_m = torch.ones(bs, m, 1).float().to(device)
        one_n = torch.ones(bs, n, 1).float().to(device)

        A = torch.bmm(torch.bmm(C1 ** 2, torch.bmm(T, one_n)), torch.transpose(one_n, 1, 2))
        B = torch.bmm(one_m, torch.transpose(torch.bmm(C2 ** 2, torch.bmm(torch.transpose(T, 1, 2), one_m)), 1, 2))
        C = torch.bmm(torch.bmm(C1, T), torch.transpose(2 * C2, 1, 2))

        return A + B - C

    @staticmethod
    def solve_gromov_linesearch(T, dT, Cc, C1, C2, M, alpha, costG):
        """_summary_

        Parameters
        ----------
        T : torch.Tensor
            The transport plan
            Shape (bs, m, n)
        dT : torch.Tensor
            _description_
        Cc : torch.Tensor
            ???
            Shape (bs, m, n)
        C1 : torch.Tensor
            The internal costs matrix between the nodes of the first graph
            Shape (bs, m, m)
        C2 : torch.Tensor
            The internal costs matrix between the nodes of the first graph
            Shape (bs, n, n)
        M : torch.Tensor
            The external costs matrix between the nodes of the two graphs
            Shape (bs, m, n)
        alpha : float
            ???

        Returns
        -------
        float
            tau
        """
        LxdT = LossFunction.tensor_product2(C1, C2, dT)
        a = (1 - alpha) * torch.sum(torch.sum(LxdT * dT, dim=-1), dim=-1)
        b = alpha * torch.sum(torch.sum(M * dT, dim=-1), dim=-1) \
            + (1 - alpha) * (torch.sum(torch.sum(LxdT * T, dim=-1), dim=-1)
                             + torch.sum(torch.sum((Cc - torch.bmm(2 * torch.bmm(C1, T), torch.transpose(C2, 1, 2)))
                                                   * dT, dim=-1), dim=-1))

        # dot = torch.bmm(C1, torch.bmm(dT, torch.transpose(C2, 1, 2)))
        # a = - 2 * (1-alpha) * torch.sum(dot * dT, dim=(1, 2))
        # b =  alpha * torch.sum(M * dT, dim=(1,2)) - 2 * (1-alpha) * (torch.sum(dot * T, dim=(1,2)) + torch.sum(torch.bmm(C1, torch.bmm(T, torch.transpose(C2, 1, 2))) * dT, dim=(1,2)))

        taus = LossFunction.batch_solve_1d_linesearch_quad(a, b)
        newcost = costG + torch.sum(a*(taus**2)+b*taus)
        return taus, newcost

    @staticmethod
    def batch_solve_1d_linesearch_quad(a, b):
        taus = []
        device = a.device
        for aa, bb in zip(a, b):
            taus.append(LossFunction.solve_1d_linesearch_quad(aa, bb))
        return torch.Tensor(taus).to(device)

    @staticmethod
    def solve_1d_linesearch_quad(a, b):
        if a > 0:  # convex
            minimum = min(1., max(0., -b / (2.0 * a)))
            return minimum
        else:  # non convex
            if a + b < 0:
                return 1.
            else:
                return 0.

    @staticmethod
    def OT_batch2(M, beta=0.5, iteration=50):
        """_summary_

        Parameters
        ----------
        M : torch.Tensor
            The external costs matrix between the nodes of the two graphs
            Shape (bs, m, n)
        beta : float
            The entropic regularisation coefficient
        iteration : int
            The number of iteration

        Returns
        -------
        torch.Tensor
            The optimal transport plan
        """
        bs, m, n = M.shape
        u = (torch.ones(bs, int(m), 1).float() / float(m)).to(M.device)
        v = (torch.ones(bs, int(n), 1).float() / float(n)).to(M.device)
        K = torch.exp(-M / beta).float()
        for t in range(iteration):
            KtransposeU = torch.bmm(torch.transpose(K, 1, 2), u)
            prevu = u
            prevv = v
            v = 1 / (n * KtransposeU)
            u = 1 / (m * torch.bmm(K, v))

            if (torch.any(KtransposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isinf(u))
                or torch.any(torch.isnan(v)) or torch.any(torch.isinf(v))
            ):
                print(f"Numerical errors at iteration {t} in Sinkhorn's algorithm. Keeping previous values")
                u = prevu
                v = prevv
                break
            
            if t % 10 == 9:
                if torch.linalg.vector_norm(u - prevu) + torch.linalg.vector_norm(v - prevv) < 1e-7:
                    break
        T = u * K * torch.transpose(v, 2, 1)
        return T

    @staticmethod
    def cost_matrix_batch(x, y, tau=0.5):
        """
          x: (bs, d, m)
          y: (bs, d, n)
          returns:
            cos_dis: (bs, n, m)
        """
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)

        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)
        cos_dis = torch.exp(- cos_dis / tau)
        return cos_dis #cos_dis.transpose(2, 1)

    @staticmethod
    def cos_batch(x, y, tau):
        bs = x.size(0)
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)
        cos_dis = torch.exp(- cos_dis / tau)

        beta = 0.1
        min_score = cos_dis.view(bs, -1).min(dim=-1)[0]
        max_score = cos_dis.view(bs, -1).max(dim=-1)[0]
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold.view(bs, 1, 1)
        return torch.nn.functional.relu(res)