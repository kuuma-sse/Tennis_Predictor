import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import *
import matplotlib.pyplot as plt
from tqdm import tqdm



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




if __name__ == '__main__':
    train_data = torch.load('train_data.pt', weights_only=False)
    val_data = torch.load('val_data.pt', weights_only=False)
    test_data = torch.load('test_data.pt', weights_only=False)

    node_num_features = train_data.x.shape[1]
    model = GraphSageTrain(in_channels=node_num_features, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

# model training
    progress_bar = tqdm(range(300), bar_format="\033[92m{l_bar}{bar}{r_bar}\033[0m")

    for epoch in progress_bar:
        actual_epoch = epoch + 1



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

            probs = torch.sigmoid(val_out).cpu().numpy()
            predicts = (probs > 0.5).astype(int)
            labels = val_data.edge_label.cpu().numpy()

            acc = accuracy_score(labels, predicts)
            precision = precision_score(labels, predicts)
            recall = recall_score(labels, predicts)
            f1 = f1_score(labels, predicts)
            auc = roc_auc_score(labels, probs)


        if epoch % 10 == 9:
            start = (epoch // 10) * 10 + 1
            end = epoch + 1
            tqdm.write(f"Epochs {start}-{end} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # testing
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
        test_loss = criterion(test_out, test_data.edge_label)

        probs = torch.sigmoid(test_out).cpu().numpy()
        predicts = (probs > 0.5).astype(int)
        labels = test_data.edge_label.cpu().numpy()

        acc = accuracy_score(labels, predicts)
        precision = precision_score(labels, predicts)
        recall = recall_score(labels, predicts)
        f1 = f1_score(labels, predicts)
        auc = roc_auc_score(labels, probs)

        print(f'Test Loss: {test_loss.item():.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Test Accuracy: {acc:.4f}')
        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test F1: {f1:.4f}')
        print(f'Test AUC: {auc:.4f}')

        # confusion matrix
        cm = confusion_matrix(labels, predicts)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Opponent wins", "Player wins"])
        display.plot()
        plt.title("Confusion Matrix - Graph Sage Test Set")
        plt.savefig('ConfusionMatrix_GraphSageTestSet.png')
        plt.show()

    # model save

    torch.save(model.state_dict(), 'GraphSage_Model_v3.pt')
    print('Graph Sage Model Version 3 saved')