import pandas as pd
import numpy as np
import streamlit as st
from nba_api.stats.endpoints import playergamelog, commonallplayers
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import time

# --- Config Streamlit ---
st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")

# --- Dictionnaire des équipes actuelles (à mettre à jour manuellement pour 2025-26) ---
current_teams = {
    "Mo Bamba": "Agent libre"  # À ajuster selon les signatures récentes
}

# --- 1. Chargement des joueurs ---
@st.cache_data(show_spinner="Chargement des joueurs…")
def load_players():
    csv_file = "players.csv"
    try:
        df = pd.read_csv(csv_file)
        if not df.empty:
            return df
    except FileNotFoundError:
        pass

    # Téléchargement une seule fois si CSV inexistant
    try:
        players = commonallplayers.CommonAllPlayers().get_data_frames()[0]
        df = players[["PERSON_ID", "DISPLAY_FIRST_LAST"]].dropna()
        df.to_csv(csv_file, index=False)
        return df
    except Exception as e:
        st.error(f"Impossible de récupérer la liste des joueurs : {e}")
        return pd.DataFrame(columns=["PERSON_ID", "DISPLAY_FIRST_LAST"])

# --- 2. Récupération des logs de matchs ---
@st.cache_data(show_spinner="Récupération des matchs…")
def get_player_game_logs(player_id, season="2024-25", max_games=50):
    try:
        logs = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=30).get_data_frames()[0]
        logs = logs.sort_values("GAME_DATE", ascending=True).tail(max_games)
        return logs
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        return pd.DataFrame()

# --- 3. Préparation des features ---
def prepare_player_stats(logs):
    if logs.empty:
        return pd.DataFrame()

    stats = logs[["GAME_DATE", "PTS", "REB", "AST", "MIN", "PLUS_MINUS", "FG_PCT", "TOV"]].copy()
    stats["GAME_DATE"] = pd.to_datetime(stats["GAME_DATE"])
    stats = stats.sort_values("GAME_DATE")

    for col in ["PTS", "REB", "AST", "MIN", "PLUS_MINUS", "FG_PCT", "TOV"]:
        stats[f"{col}_MA5"] = stats[col].rolling(window=5, min_periods=1).mean()

    targets = ["PTS", "REB", "AST", "MIN", "FG_PCT", "TOV"]
    stats[[f"NEXT_{t}" for t in targets]] = stats[targets].shift(-1)

    return stats.dropna()

# --- 4. Modèle prédictif ---
@st.cache_resource(show_spinner="Entraînement du modèle…")
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def simulate_season(stats, model, num_games=82, min_minutes=15):
    if stats.empty:
        return pd.DataFrame()

    features = ["PTS_MA5", "REB_MA5", "AST_MA5", "MIN_MA5", "PLUS_MINUS_MA5", "FG_PCT_MA5", "TOV_MA5"]
    last_features = stats[features].iloc[-1].values

    rmse = np.sqrt(mean_squared_error(
        stats[[f"NEXT_{f.split('_')[0]}" for f in features[:-1]]].values,
        model.predict(stats[features].values)
    ))

    predictions = []
    current_features = last_features.copy()

    for _ in range(num_games):
        pred = model.predict(current_features.reshape(1, -1))[0]
        pred += np.random.normal(0, rmse)
        pred[3] = max(pred[3], min_minutes)   # Minutes minimales
        pred[0] = max(pred[0], pred[3] * 0.5) # Points >= 50% des minutes
        pred[1] = max(pred[1], pred[3] * 0.4) # Rebonds >= 40% des minutes
        pred[2] = max(pred[2], pred[3] * 0.02)# Passes >= 2% des minutes
        pred[4] = min(max(pred[4], 0.4), 0.6) # FG% borné
        pred[5] = max(pred[5], 0.5)           # TOV borné

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
selected_player = st.sidebar.selectbox("Choisir un joueur", options=list(player_dict.keys()))
player_id = player_dict[selected_player]
max_games = st.sidebar.slider("Nombre de matchs à analyser", min_value=10, max_value=100, value=50)

logs = get_player_game_logs(player_id, max_games=max_games)
stats = prepare_player_stats(logs)

if not stats.empty:
    st.subheader(f"Performances de {selected_player}")
    st.write(f"**Équipe actuelle :** {current_teams.get(selected_player, 'Non confirmée pour 2025-26')}")

    features = ["PTS_MA5", "REB_MA5", "AST_MA5", "MIN_MA5", "PLUS_MINUS_MA5", "FG_PCT_MA5", "TOV_MA5"]
    targets = ["NEXT_PTS", "NEXT_REB", "NEXT_AST", "NEXT_MIN", "NEXT_FG_PCT", "NEXT_TOV"]

    X, y = stats[features].values, stats[targets].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    st.write(f"**Erreur quadratique moyenne (MSE) :** PTS: {mse[0]:.2f}, REB: {mse[1]:.2f}, AST: {mse[2]:.2f}, MIN: {mse[3]:.2f}, FG_PCT: {mse[4]:.2f}, TOV: {mse[5]:.2f}")

    # Simulation pour la saison 2025-26
    season_sim = simulate_season(stats, model, min_minutes=15)
    st.subheader("Simulation pour la saison 2025-26")
    st.write(f"Moyenne prédite : PTS: {season_sim['PTS'].mean():.2f}, REB: {season_sim['REB'].mean():.2f}, AST: {season_sim['AST'].mean():.2f}, "
             f"MIN: {season_sim['MIN'].mean():.2f}, FG_PCT: {season_sim['FG_PCT'].mean():.2f}, TOV: {season_sim['TOV'].mean():.2f}")

    # Graphique : réels vs prédits pour PTS
    fig1 = px.scatter(x=list(range(len(y_test))), y=y_test[:, 0], labels={'x': 'Match', 'y': 'Points Réels'}, title="Points Réels vs Prédits")
    fig1.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, 0], mode='markers', name='Points Prédits')
    st.plotly_chart(fig1)

    # Courbe des points + moyenne mobile
    fig2 = px.line(stats, x="GAME_DATE", y=["PTS", "PTS_MA5"], title="Points & Moyenne Mobile")
    st.plotly_chart(fig2)

    # Graphiques optionnels
    if st.checkbox("Afficher les graphiques pour REB, AST, MIN, FG_PCT, TOV"):
        for idx, col in enumerate(["REB", "AST", "MIN", "FG_PCT", "TOV"], start=1):
            fig = px.scatter(x=list(range(len(y_test))), y=y_test[:, idx], labels={'x': 'Match', 'y': f'{col} Réels'}, title=f"{col} Réels vs Prédits")
            fig.add_scatter(x=list(range(len(y_pred))), y=y_pred[:, idx], mode='markers', name=f'{col} Prédits')
            st.plotly_chart(fig)
else:
    st.warning("Pas de données disponibles pour ce joueur.")
