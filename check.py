
import torch as nn
import pandas as pd
import glob
import matplotlib as plt
import numpy as np
from torch_geometric.data import Data

from numpy.ma.extras import average
from six import print_

path = r'C:\Users\hkute\Documents\LearningPython\tennis_predictor\Tennis_Predictor\datasets'

all_files = glob.glob(path + r"\atp_matches_*.csv")

df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

pd.set_option('display.max_columns', None)

# one hot encoding surface
surface = pd.get_dummies(df.surface, dtype=int)

df = pd.concat([df, surface], axis='columns')
df.drop(["surface"], axis='columns', inplace=True)

# separating data to historical and future data
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

historical_df = df[df['tourney_date'] < "2023-01-01"]
future_df = df[df['tourney_date'] >= "2023-01-01"]

# separate winner stats from losers stats
winner_stats = [col for col in historical_df.columns if col.startswith('w_')]
loser_stats = [col for col in historical_df.columns if col.startswith('l_')]

# create winner and loser df
winners = historical_df[['winner_name'] + winner_stats].copy()
winners.columns = ['player'] + [col[2:] for col in winner_stats]

losers = historical_df[['loser_name'] + loser_stats].copy()
losers.columns = ['player'] + [col[2:] for col in loser_stats]

# combine results
players_df = pd.concat([winners, losers], ignore_index=True)

# get average
player_averages = players_df.groupby('player').mean()

player_averages = player_averages.reset_index()

atp_players = pd.read_csv(r'datasets\atp_players.csv', dtype={'wikidata_id':'string'})
#print(atp_players)

atp_players_new = atp_players.copy()

atp_players_new['player'] = atp_players_new['name_first'] +' '+ atp_players_new['name_last']
#print(atp_players_new)


official_df = player_averages.merge(atp_players_new[['player','height','player_id','dob','hand','ioc']], on='player', how='left')
official_df['player_id'] = official_df['player_id'].astype('Int64')
official_df['dob'] = official_df['dob'].astype('Int64')

#print(official_df)
#specific_player = official_df[official_df['ace'] > 15]
#print(specific_player)

print(max(official_df['player_id'].value_counts()))

#sort players and extract node features

sorted_df = official_df.sort_values(by='player_id')

node_features = sorted_df[['ace','df','svpt','1stIn','1stWon','2ndWon', 'SvGms','bpSaved','bpFaced','height','dob','hand']]

#convert non-numeric vals

pd.set_option('mode.chained_assignment', None)
#handness = node_features["hand"].str.split(",", expand=True)
hands = pd.get_dummies(node_features.hand, dtype=int)

node_features = pd.concat([node_features, hands], axis='columns')
node_features.drop(["hand"], axis='columns', inplace=True)

#print(node_features)

# Convert to numpy
x = node_features.to_numpy()
print(x.shape) # [num_nodes x num_features]


# node index for players to be recognised by row
sorted_df = sorted_df.reset_index(drop=True)

player_id_to_idx = {
    player_id: idx
    for idx, player_id in enumerate(sorted_df['player_id'])
}

#print(player_id_to_idx)

# creating edges for graph

edge_index_np = np.stack([historical_df['winner_id'].values, historical_df['loser_id'].values], axis=0)
edge_index = nn.tensor(edge_index_np, dtype=nn.long)
#edge_index = nn.tensor([historical_df['winner_id'].values, historical_df['loser_id'].values], dtype=nn.long)
print(edge_index.shape)

# creating edge attributes

edge_attr = nn.tensor(historical_df[['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                                     'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
                                     'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
                                     'Carpet', 'Clay', 'Grass', 'Hard']].values, dtype=nn.float)


data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


#print(historical_df.columns.tolist())