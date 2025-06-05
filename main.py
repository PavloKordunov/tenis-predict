from src.preprocessing.load_data import load_raw_matches
from src.features.build_features import build_features
from src.models.train_model import train_and_evaluate
from src.evaluation.predict_match import predict_single_match
import joblib
from pathlib import Path

def main():
    df_raw = load_raw_matches("data/raw/")
    df_features = build_features(df_raw)
    
    model = train_and_evaluate(df_features)
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/train_model.pkl")
    
    match_to_predict = {
        'player1_name': 'Jannik Sinner',
        'player1_rank': 3,
        'player1_age': 23,
        'player1_country': 'ITA',
        'player2_name': 'Alexander Zverev',
        'player2_rank': 5,
        'player2_age': 28,
        'player2_country': 'GER',
        'surface': 'Hard',
        'date': 20250820,
        'tourney_id': 'USO2025'
    }

   
    prediction = predict_single_match(
        model_path="models/train_model.pkl",
        match_info=match_to_predict,
        historical_data_df=df_raw
    )
    
    print("\nРезультат прогнозу:")
    print(f"Ймовірність перемоги {match_to_predict['player1_name']}: {prediction['player1_win_probability']:.1%}")
    print(f"Ймовірність перемоги {match_to_predict['player2_name']}: {prediction['player2_win_probability']:.1%}")
    print(f"Прогнозований переможець: {prediction['predicted_winner']}")
    print(f"Рівень впевненості: {prediction['confidence']:.1%}")

if __name__ == "__main__":
    main()
