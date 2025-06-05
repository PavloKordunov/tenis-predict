import joblib
import pandas as pd
from src.features.build_features import build_features
from datetime import datetime

def predict_single_match(model_path: str, match_info: dict, historical_data_df: pd.DataFrame = None, historical_data_path: str = None):
    model = joblib.load(model_path)

    if historical_data_df is not None:
        historical_data = historical_data_df
    elif historical_data_path is not None:
        historical_data = pd.read_csv(historical_data_path)
    else:
        raise ValueError("Either historical_data_df or historical_data_path must be provided.")

    match_date = pd.to_datetime(match_info.get('date', datetime.today().strftime('%Y%m%d')), format='%Y%m%d')

    match_data = pd.DataFrame([{
        'tourney_date': match_date,
        'surface': match_info['surface'],
        'winner_name': match_info['player1_name'],
        'loser_name': match_info['player2_name'],
        'winner_rank': match_info['player1_rank'],
        'loser_rank': match_info['player2_rank'],
        'winner_age': match_info['player1_age'],
        'loser_age': match_info['player2_age'],
        'winner_ioc': match_info.get('player1_country', 'UNK'),
        'loser_ioc': match_info.get('player2_country', 'UNK'),
        'tourney_id': match_info.get('tourney_id', 'UNKNOWN')
    }])

    combined_data = pd.concat([historical_data, match_data], ignore_index=True)
    df_processed = build_features(combined_data).iloc[-2:]
    feature_columns = [
        "rank_diff", "age_diff", "surface_hard", "surface_clay", "surface_grass",
        "elo_diff", "h2h_winrate", "recent_winrate_diff", "same_country", "home_advantage"
    ]

    probs = model.predict_proba(df_processed[feature_columns])

    if probs[0][1] > probs[1][1]:
        predicted_winner = match_info['player1_name']
        player1_proba = probs[0][1]
        player2_proba = 1 - player1_proba
    else:
        predicted_winner = match_info['player2_name']
        player2_proba = probs[1][1]
        player1_proba = 1 - player2_proba

    confidence = max(player1_proba, player2_proba)

    return {
        "player1_win_probability": round(player1_proba, 1),
        "player2_win_probability": round(player2_proba, 1),
        "predicted_winner": predicted_winner,
        "confidence": round(confidence, 1)
    }
