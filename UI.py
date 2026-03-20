import streamlit as st
import pandas as pd
from inference import predict, match_prediction_xgb

df = pd.read_csv(r'C:\Users\hkute\Documents\LearningPython\tennis_predictor\Tennis_Predictor\datasets\atp_players.csv')

df.drop(columns=['wikidata_id'], inplace=True)
df['player'] = df['name_first'] + ' ' + df['name_last']

st.write("""
# GraphSage & XGBoost Tennis Match Preditor 

""")

player1 = st.selectbox(
    "Choose Player 1",
    df['player'],
    key="player1"
)



player2 = st.selectbox(
    "Choose a Player 2",
    df['player'],
    key="player2"
)


player1_id = df.loc[df["player"] == player1, "player_id"].values[0]
player2_id = df.loc[df["player"] == player2, "player_id"].values[0]

st.write(player1)
st.write(player2)

if st.button("Predict Match"):
    graph_sage_prob = predict(player1_id, player2_id)
    xgb_prob = match_prediction_xgb(player1_id, player2_id)

    st.subheader("Graph Sage Predicted Match")
    st.write(f"{player1} win probability: {graph_sage_prob:.2%}")
    st.write(f"{player2} win probability: {1 - graph_sage_prob:.2%}")

    st.subheader("XGBoost Predicted Match")
    st.write(f"{player1} win probability: {xgb_prob:.2%}")
    st.write(f"{player2} win probability: {1 - xgb_prob:.2%}")

print(player1_id)
print(player2_id)