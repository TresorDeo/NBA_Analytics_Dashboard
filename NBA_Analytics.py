import pandas as pd
import numpy as np
import streamlit as st
import requests
import os

try:
    from nba_api.stats.endpoints import playergamelog, commonallplayers
except ImportError as e:
    st.error("Error: 'nba_api' is not installed. Please ensure 'nba_api' is in requirements.txt and redeploy. Details: " + str(e))
    st.stop()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px

# --- Config Streamlit ---
st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")

# --- Dictionnaire des équipes actuelles ---
current_teams = {
    "Mo Bamba": "Free Agent"
    # Add other players as needed
}

# --- 1. Chargement des joueurs ---
@st.cache_data
def load_players():
    st.write("Loading players data...")
    try:
        # Vérifie si le fichier CSV existe déjà
        if os.path.exists("players.csv"):
            st.write("Using cached players.csv")
            return pd.read_csv("players.csv")
        # Sinon, récupère les données avec un délai d'attente plus long
        st.write("Fetching data from NBA API...")
        players = commonallplayers.CommonAllPlayers(timeout=120).get_data_frames()[0]  # Timeout de 120 secondes
        df = players[["PERSON_ID", "DISPLAY_FIRST_LAST"]].dropna()
        df.to_csv("players.csv", index=False)
        st.write("Data fetched and cached successfully")
        return df
    except requests.exceptions.ReadTimeout:
        st.error("Timeout error: The NBA API took too long to respond. Please try again later or check your internet connection.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading players data: {str(e)}. Please ensure the NBA API is accessible.")
        st.stop()

# --- 2. Logs d’un joueur ---
@st.cache_data
def get_player_game_logs(player_id, season="2024-25", max_games=50):
    st.write(f"Loading game logs for player ID {player_id}...")
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=120).get_data_frames()[0]
    logs = logs.sort_values("GAME_DATE", ascending=True).tail(max_games)
    st.write("Game logs loaded successfully")
    return logs

# --- 3. Préparation des features ---
def prepare_player_stats(logs):
    if logs.empty:
        return pd.DataFrame()
    stats = logs[["GAME_DATE", "PTS", "REB", "AST", "MIN", "PLUS_MINUS", "FG_PCT", "TOV"]].copy()
    stats["GAME_DATE"] = pd.to_datetime(stats["GAME_DATE"])
    stats = stats.sort_values("GAME_DATE")

    for col in ["PTS", "REB", "AST", "MIN", "PLUS_MINUS", "FG_PCT", "TOV"]:
        stats[f"{col}_MA5"] = stats[col].rolling(window=5, min_periods=1).mean()

    stats[["NEXT_PTS", "NEXT_REB", "NEXT_AST", "NEXT_MIN", "NEXT_FG_PCT", "NEXT_TOV"]] = stats[["PTS", "REB", "AST", "MIN", "FG_PCT", "TOV"]].shift(-1)
    return stats.dropna()

# --- 4. Modèle prédictif ---
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def simulate_season(stats, model, num_games=82, min_minutes=15):
    if stats.empty:
        return pd.DataFrame()
    last_features = stats[["PTS_MA5", "REB_MA5", "AST_MA5", "MIN_MA5", "PLUS_MINUS_MA5", "FG_PCT_MA5", "TOV_MA5"]].iloc[-1].values
    predictions = []
    current_features = last_features.copy()
    rmse = np.sqrt(mean_squared_error(stats[["NEXT_PTS", "NEXT_REB", "NEXT_AST", "NEXT_MIN", "NEXT_FG_PCT", "NEXT_TOV"]].values, 
                                     model.predict(stats[["PTS_MA5", "REB_MA5", "AST_MA5", "MIN_MA5", "PLUS_MINUS_MA5", "FG_PCT_MA5", "TOV_MA5"]].values)))

    for _ in range(num_games):
        pred = model.predict(current_features.reshape(1, -7))[0]
        pred += np.random.normal(0, rmse)
        pred[3] = max(pred[3], min_minutes)
        pred[0] = max(pred[0], pred[3] * 0.5)
        pred[1] = max(pred[1], pred[3] * 0.4)
        pred[2] = max(pred[2], pred[3] * 0.02)
        pred[4] = min(max(pred[4], 0.4), 0.6)
        pred[5] = max(pred[5], 0.5)
        predictions.append(pred)
        current_features = np.roll(current_features, -6)
        current_features[-6:] = pred[:6]

    return pd.DataFrame(predictions, columns=["PTS", "REB", "AST", "MIN", "FG_PCT", "TOV"])

# --- 5. Interface ---
st.title("🏀 NBA Analytics Dashboard")

players_df = load_players()
player_dict = {row["DISPLAY_FIRST_LAST"]: row["PERSON_ID"] for _, row in players_df.iterrows()}

# Sidebar
st.sidebar.title("Options")
selected_player = st.sidebar.selectbox("Select a player", options=list(player_dict.keys()))
player_id = player_dict[selected_player]
max_games = st.sidebar.slider("Number of games to analyze", min_value=10, max_value=100, value=50)

logs = get_player_game_logs(player_id, max_games=max_games)
stats = prepare_player_stats(logs)

if not stats.empty:
    st.subheader(f"Performance of {selected_player}")
    st.write(f"**Current Team:** {current_teams.get(selected_player, 'Unconfirmed for 2025-26')}")

    features = ["PTS_MA5", "REB_MA5", "AST_MA5", "MIN_MA5", "PLUS_MINUS_MA5", "FG_PCT_MA5", "TOV_MA5"]
    targets = ["NEXT_PTS", "NEXT_REB", "NEXT_AST", "NEXT_MIN", "NEXT_FG_PCT", "NEXT_TOV"]
    X = stats[features].values
    y = stats[targets].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")

    st.write(f"**Mean Squared Error (MSE):** PTS: {mse[0]:.2f}, REB: {mse[1]:.2f}, AST: {mse[2]:.2f}, MIN: {mse[3]:.2f}, FG_PCT: {mse[4]:.2f}, TOV: {mse[5]:.2f}")

    # Simulation for 2025-26 season
    season_sim = simulate_season(stats, model, min_minutes=15)
    st.subheader("Simulation for 2025-26 Season")
    st.write(f"Predicted Average: PTS: {season_sim['PTS'].mean():.2f}, REB: {season_sim['REB'].mean():.2f}, AST: {season_sim['AST'].mean():.2f}, "
             f"MIN: {season_sim['MIN'].mean():.2f}, FG_PCT: {season_sim['FG_PCT'].mean():.2f}, TOV: {season_sim['TOV'].mean():.2f}")

    # Graph: Actual vs Predicted for PTS
    fig1 = px.scatter(
        x=list(range(len(y_test))),
        y=y_test[:, 0],
        labels={'x': 'Match', 'y': 'Actual Points'},
        title="Actual vs Predicted Points"
    )
    fig1.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 0], mode='markers', name='Predicted Points')
    st.plotly_chart(fig1)

    # Line chart: Points + Moving Average
    fig2 = px.line(stats, x="GAME_DATE", y=["PTS", "PTS_MA5"], title="Points & Moving Average")
    st.plotly_chart(fig2)

    # Optional graphs
    if st.checkbox("Show graphs for REB, AST, MIN, FG_PCT, TOV"):
        fig_reb = px.scatter(x=list(range(len(y_test))), y=y_test[:, 1], labels={'x': 'Match', 'y': 'Actual Rebounds'}, title="Actual vs Predicted Rebounds")
        fig_reb.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 1], mode='markers', name='Predicted Rebounds')
        st.plotly_chart(fig_reb)

        fig_ast = px.scatter(x=list(range(len(y_test))), y=y_test[:, 2], labels={'x': 'Match', 'y': 'Actual Assists'}, title="Actual vs Predicted Assists")
        fig_ast.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 2], mode='markers', name='Predicted Assists')
        st.plotly_chart(fig_ast)

        fig_min = px.scatter(x=list(range(len(y_test))), y=y_test[:, 3], labels={'x': 'Match', 'y': 'Actual Minutes'}, title="Actual vs Predicted Minutes")
        fig_min.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 3], mode='markers', name='Predicted Minutes')
        st.plotly_chart(fig_min)

        fig_fg_pct = px.scatter(x=list(range(len(y_test))), y=y_test[:, 4], labels={'x': 'Match', 'y': 'Actual FG_PCT'}, title="Actual vs Predicted FG_PCT")
        fig_fg_pct.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 4], mode='markers', name='Predicted FG_PCT')
        st.plotly_chart(fig_fg_pct)

        fig_tov = px.scatter(x=list(range(len(y_test))), y=y_test[:, 5], labels={'x': 'Match', 'y': 'Actual TOV'}, title="Actual vs Predicted TOV")
        fig_tov.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 5], mode='markers', name='Predicted TOV')
        st.plotly_chart(fig_tov)

else:
    st.warning("No data available for this player.")
