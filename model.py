# This is a toy model followed Pytorch Geometric
# download dataset, build model, train model and finally test model
# implement two layer GCN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# download dataset
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./Data', name='Cora')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input channel: how many features a node has
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        # each node in this graph has 1443 features, and 2708 nodes
        # and they belong to 7 classes
        # output channel: how many output channel a graph has
        self.conv2 = GCNConv(16, dataset.num_classes)
        # conclustion: when we define network, we just define how many node features and how output channels, we do not
        #  define how many nodes in a graph etc.

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# set model to train mode, epoch 200
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# set model to evaluation mode
model.eval()
# we just want index value
_, pred = model(data).max(dim=1)
# data.test_mask is a tensor of True or False, it pick up some data as test dataset, and the number of test_dataset is
# 1000, we could print shape to get number
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
# finally we get accuracy is 0.8050
print('Accuracy:{:.4f}'.format(acc))