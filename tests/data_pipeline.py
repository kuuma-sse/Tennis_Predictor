import pytest
import torch
from torch_geometric.data import Data

@pytest.fixture
def train_data():
    return torch.load(r'C:\Users\hkute\Documents\LearningPython\tennis_predictor\Tennis_Predictor\train_data.pt', weights_only=False)

@pytest.fixture
def val_data():
    return torch.load(r'C:\Users\hkute\Documents\LearningPython\tennis_predictor\Tennis_Predictor\val_data.pt', weights_only=False)

@pytest.fixture
def test_data():
    return torch.load(r'C:\Users\hkute\Documents\LearningPython\tennis_predictor\Tennis_Predictor\test_data.pt', weights_only=False)


class TestDataStruc:

    def test_train_is_real(self, train_data):
        assert isinstance(train_data, Data)

    def test_val_is_real(self, val_data):
        assert isinstance(val_data, Data)

    def test_test_is_real(self, test_data):
        assert isinstance(test_data, Data)

    def test_train_has_needed_attributes(self, train_data):
        assert hasattr(train_data, 'x')
        assert hasattr(train_data, 'edge_index')
        assert hasattr(train_data, 'edge_label')
        assert hasattr(train_data, 'edge_label_index')


class TestNodeFeatures:

    def test_train_feature_dimensions(self, train_data):
        assert train_data.x.shape[1] == 19

    def test_val_feature_dimensions(self, val_data):
        assert val_data.x.shape[1] == 19

    def test_test_feature_dimensions(self, test_data):
        assert test_data.x.shape[1] == 19

    def test_no_nan_vals_in_train_feat(self, train_data):
        assert not torch.isnan(train_data.x).any().item()

    def test_no_nan_vals_in_val_feat(self, val_data):
        assert not torch.isnan(val_data.x).any().item()

    def test_no_nan_vals_in_test_feat(self, test_data):
        assert not torch.isnan(test_data.x).any().item()

    def test_no_inf_val_in_train_feat(self, train_data):
        assert not torch.isinf(train_data.x).any().item()


class TestEdgeIndex:

    def test_index_dimensions(self, train_data):
        assert train_data.edge_index.shape[0] == 2

    def test_edge_index_within_bounds(self, train_data):
        """All node indices in edge_index should be within bounds"""
        num_nodes = train_data.x.shape[0]
        assert train_data.edge_index.max() < num_nodes
        assert train_data.edge_index.min() >= 0

    def test_edge_index_dtype(self, train_data):
        """All edge indices in edge_index should have the same dtype"""
        assert train_data.edge_index.dtype == torch.long