import torch
import joblib
import json
import numpy as np
import pandas as pd
from streamlit import columns

from GraphSageTrain import GraphSageTrain

# Graph Sage Inference
train_data = torch.load('train_data.pt', weights_only=False)
num_node_features = train_data.x.shape[1]
model = GraphSageTrain(in_channels=num_node_features,hidden_channels=64)
model.load_state_dict(torch.load('GraphSage_Model_v2.pt', weights_only=False))
player_id_to_idx = torch.load('player_id_to_idx.pt', weights_only=False)
model.eval()

def predict(player_1_id,player_2_id):

    player_1_idx = player_id_to_idx[player_1_id]
    player_2_idx = player_id_to_idx[player_2_id]

    # predict A vs B
    edge_ab = torch.tensor([[player_1_idx], [player_2_idx]], dtype=torch.long)
    # predict B vs A
    edge_ba = torch.tensor([[player_2_idx], [player_1_idx]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        prob_ab = torch.sigmoid(model(train_data.x, train_data.edge_index, edge_ab)).item()
        prob_ba = torch.sigmoid(model(train_data.x, train_data.edge_index, edge_ba)).item()

    # average both directions for a fair prediction
    prob_a_wins = (prob_ab + (1 - prob_ba)) / 2
    return prob_a_wins


# XGBoost Inference

# load xgboost and feature columns

xgb = joblib.load('xgb_model.pkl')
with open('xgb_features.json') as f:
    xgb_feature_cols = json.load(f)

official_df = pd.read_csv('official_df.csv')

#searching dataframe for players averages
def match_prediction_xgb(player_1_id,player_2_id):

    player = official_df[official_df['player_id'] == player_1_id]
    opponent = official_df[official_df['player_id'] == player_2_id]

    if player.empty or opponent.empty:
        return None


    player_avg = player.add_prefix('avg_player_').reset_index(drop=True)
    opponent_avg = opponent.add_prefix('avg_opponent_').reset_index(drop=True)

    player_avg = player_avg.rename(columns={'avg_player_player_id': 'player_id'})
    opponent_avg = opponent_avg.rename(columns={'avg_opponent_player_id': 'opponent_id'})



    row = pd.concat([player_avg, opponent_avg], axis=1)

    row = row.reindex(columns=xgb_feature_cols, fill_value=0)

    prob = xgb.predict_proba(row)[0][1]
    return prob


