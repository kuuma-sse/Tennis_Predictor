import pytest
import torch
from torch_geometric.data  import Data
from model import GraphSageModel

@pytest.fixture
def sample_graph():

    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]])
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def model():
    return GraphSageModel(in_channels=8, hidden_channels=32, out_channels=2)







