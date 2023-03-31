import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl
from dgl.nn.pytorch import GMMConv
from dgl.data import FlickrDataset

class MoNet(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 dim,
                 n_kernels,
                 dropout):
        super(MoNet, self).__init__()
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            # GMMConv(in_feats, n_hidden, dim, n_kernels))
            GMMConv(in_feats, n_hidden, dim, n_kernels, allow_zero_in_degree=True))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(GMMConv(n_hidden, out_feats, dim, n_kernels, allow_zero_in_degree=True))
        # self.layers.append(GMMConv(n_hidden, out_feats, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, feat, middle=False):
        middle_feats = []
        h = feat

        for i in range(len(self.layers)):
            # blocks[i] = dgl.add_self_loop(blocks[i])
            us, vs = blocks[i].edges(order='eid')
            udeg, vdeg = 1 / torch.sqrt(blocks[i].in_degrees(us).float()), 1 / torch.sqrt(blocks[i].in_degrees(vs).float())
            pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](
                blocks[i], h, self.pseudo_proj[i](pseudo))
            if i != len(self.layers)-1:
                middle_feats.append(h)
        if middle:
            return h, middle_feats
        return h

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels


device = torch.device("cuda:7")
dataset = FlickrDataset()
g = dataset[0]
g = g.to(device)
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
val_and_test = (val_mask + test_mask).cpu().numpy()
train_mask = np.ones(g.ndata['feat'].shape[0]) - val_and_test
train_mask = torch.tensor(train_mask).to(test_mask)
train_index = torch.nonzero(train_mask==1).squeeze()
in_size = g.ndata['feat'].shape[1]
out_size = dataset.num_classes

model = MoNet(in_feats=in_size,
                      n_hidden=64,
                      out_feats=out_size,
                      dim=5,
                      n_kernels=5,
                      dropout=0).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

train_dataloader = dgl.dataloading.DataLoader(
                g,
                train_index,
                sampler,
                batch_size=2048,
                shuffle=True,
                drop_last=False,
                num_workers=0)
labels = g.ndata['label'].to(device)
feats =  g.ndata['feat'].to(device)
for epoch in range(800):
    model.train()
    total_loss = 0
    # mini-batch loop
    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
        blocks = [blk.to(device) for blk in blocks]
        
        batch_inputs, batch_labels = load_subtensor(feats, labels, seeds, input_nodes)

        output = model(blocks, batch_inputs)


        # loss = loss_fcn(output, labels)
        loss = F.cross_entropy(output, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch} end")
