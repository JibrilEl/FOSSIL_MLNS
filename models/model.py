import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


b_xent = nn.BCEWithLogitsLoss()

## GCN definition in the MUSE framework
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)




# Generation
class GCNGeneration(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, normalisation=F.normalize):
        super(GCNGeneration, self).__init__()
        self.conv = GCN(in_channels, out_channels)
        self.activation = activation
        self.normalisation = normalisation
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj):
        x = self.activation(self.conv(x, adj))
        return x

class GATGeneration(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, normalisation=F.normalize):
        super(GATGeneration, self).__init__()
        self.conv = GATConv(in_channels, out_channels, add_self_loops=False)
        self.activation = activation
        self.normalisation = normalisation
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj):
        x = self.activation(self.conv(x, adj))
        return x
      
# Model
class MLP(nn.Module):
    def __init__(self, in_channels, dropout_mlp, gate_channels):
        super(MLP, self).__init__()
        self.dropout = dropout_mlp

        self.gate1 = nn.Linear(in_channels, gate_channels).requires_grad_(requires_grad=True) 
        self.gate2 = nn.Linear(in_channels, gate_channels).requires_grad_(requires_grad=True) 
        self.gate3 = nn.Linear(2*gate_channels+1, 1).requires_grad_(requires_grad=True)
        self.sigm = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.gate1.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate2.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate3.weight, gain=1.414)


    def forward(self, h1, h2, deg):

        z1 = self.gate1(h1).squeeze()
        z1 = F.dropout(z1, p=self.dropout, training=self.training)


        z2 = self.gate2(h2).squeeze()
        z2 = F.dropout(z2, p=self.dropout, training=self.training)
        

        w1 = torch.cat((z1, z2, deg), dim=1)

        output = self.gate3(w1)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.sigm(output)
        return output


class Model(nn.Module):
    def __init__(self, input, hidden, drop, dropout_mlp, gate, activation):
        super(Model, self).__init__()

        self.drop = drop

        self.gen = GATGeneration(hidden, hidden, activation)
        self.fcs  = nn.ModuleList()
        self.fcs.append(GCN(input, 2 * hidden))
        self.fcs.append(GCN(2 * hidden, hidden))

        self.params = list(self.fcs.parameters()) + list(self.gen.parameters())

        self.mlp = nn.ModuleList()
        self.mlp.append(MLP(hidden, dropout_mlp, gate))
        self.params_mlp = list(self.mlp.parameters())
        self.activation = activation


    def forward(self, x, adj1, adj2, deg):
        x = F.dropout(x, p=self.drop, training=self.training)

        hs = self.activation(self.fcs[0](x, adj1))
        hs = self.fcs[1](hs, adj1)
        hs_gen = self.gen(hs, adj1)

        hc = self.activation(self.fcs[0](x, adj2))
        hc = self.fcs[1](hc, adj2)
        hc_gen = self.gen(hc, adj2)

        beta = self.mlp[0](hs.detach(), hc.detach(), deg.detach())

        beta_gen = self.mlp[0](hs_gen.detach(), hc_gen.detach(), deg.detach())
        h_gen = hs_gen + torch.mul(beta_gen.detach(), hc_gen)

        return hs, hc, h_gen, beta

    def cos_loss(self, X, N, beta):
        S = F.cosine_similarity(X, N).unsqueeze(1)
        simi = torch.matmul(beta.t(), S)
        return simi, S

    @torch.no_grad()
    def get_embedding(self, x1, adj1, adj2, deg):
        h1 = self.activation(self.fcs[0](x1, adj1))
        h1 = self.fcs[1](h1, adj1)

        h3 = self.activation(self.fcs[0](x1, adj2))
        h3 = self.fcs[1](h3, adj2)

        beta = self.mlp[0](h1.detach(), h3.detach(), deg.detach())
        return h1 + torch.mul(beta.detach(), h3)