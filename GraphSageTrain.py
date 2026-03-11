import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

train_data = torch.load('train_data.pt', weights_only=False)
val_data = torch.load('val_data.pt', weights_only=False)
test_data = torch.load('test_data.pt', weights_only=False)

node_num_features = train_data.x.shape[1]

# graph sage
class GraphSageTrain(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSageTrain, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels * 2, 1)

    def encode (self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode (self, embeddings,edge_label_index):
        src = embeddings[edge_label_index[0]]
        dst = embeddings[edge_label_index[1]]
        combined = torch.cat((src, dst), dim=1)
        return self.classifier(combined).squeeze()

    def forward(self, x, edge_index, edge_label_index):
        embeddings = self.encode(x, edge_index)
        return self.decode(embeddings, edge_label_index)





model = GraphSageTrain(in_channels=node_num_features, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# model training
for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
    loss = criterion(out, train_data.edge_label)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(val_data.x, val_data.edge_index, val_data.edge_label_index)
        val_loss = criterion(val_out, val_data.edge_label)




    if epoch % 10 == 0:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch + 1, 300, loss.item(), val_loss.item()))

# testing
model.eval()
with torch.no_grad():
    test_out = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    test_loss = criterion(test_out, test_data.edge_label)

    probs = torch.sigmoid(test_out)
    predicts = (probs > 0.5).float()
    accuracy = (predicts == test_data.edge_label).float().mean()

    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Test Accuracy: {accuracy.item():.4f}')

# model save

torch.save(model.state_dict(), 'GraphSage_Model_v2.pt')
print('Graph Sage Model Version 2 saved')