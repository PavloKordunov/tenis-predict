import pandas as pd
import numpy as np

def calculate_elo_features(df):
    K = 32
    base_rating = 1500
    player_ratings = {}

    elo_diffs = []

    df = df.sort_values("tourney_date")

    for _, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]

        winner_rating = player_ratings.get(winner, base_rating)
        loser_rating = player_ratings.get(loser, base_rating)

        expected_win = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        expected_loss = 1 - expected_win

        player_ratings[winner] = winner_rating + K * (1 - expected_win)
        player_ratings[loser] = loser_rating + K * (0 - expected_loss)

        elo_diffs.append(winner_rating - loser_rating)

    df["elo_diff"] = elo_diffs
    return df

def add_head_to_head_feature(df):
    h2h_dict = {}
    results = []
    
    for _, row in df.iterrows():
        p1 = row['winner_name'] if row['label'] == 1 else row['loser_name']
        p2 = row['loser_name'] if row['label'] == 1 else row['winner_name']
        
        wins_p1 = h2h_dict.get((p1, p2), 0)
        wins_p2 = h2h_dict.get((p2, p1), 0)
        
        total_matches = wins_p1 + wins_p2
        winrate = wins_p1 / total_matches if total_matches > 0 else 0.5
        
        results.append(winrate)
        
        if row['label'] == 1:
            h2h_dict[(p1, p2)] = wins_p1 + 1
        else:
            h2h_dict[(p2, p1)] = wins_p2 + 1
    
    df['h2h_winrate'] = results
    return df

def add_recent_winrate(df, max_matches=10):
    recent_matches = {}

    winrates = []

    for _, row in df.iterrows():
        p1 = row['winner_name'] if row['label'] == 1 else row['loser_name']
        p2 = row['loser_name'] if row['label'] == 1 else row['winner_name']

        if p1 in recent_matches:
            winrate_p1 = sum(recent_matches[p1]) / len(recent_matches[p1])
        else:
            winrate_p1 = 0.5

        if p2 in recent_matches:
            winrate_p2 = sum(recent_matches[p2]) / len(recent_matches[p2])
        else:
            winrate_p2 = 0.5

        winrates.append(winrate_p1 - winrate_p2)

        for player, result in [(p1, 1), (p2, 0)]:
            if player not in recent_matches:
                recent_matches[player] = []
            recent_matches[player].append(result)
            if len(recent_matches[player]) > max_matches:
                recent_matches[player].pop(0)

    df['recent_winrate_diff'] = winrates
    return df 

def build_features(df):
    df = df[["tourney_date", "surface", "winner_name", "loser_name", 
        "winner_rank", "loser_rank", "winner_age", "loser_age",
        "winner_ioc", "loser_ioc", "tourney_id"]].copy()

    df = df[(df["winner_rank"] > 0) & (df["loser_rank"] > 0)].dropna()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format='mixed')
    df = df.sort_values("tourney_date")
    
    df_winner = df.copy()
    df_winner["rank_diff"] = np.log1p(df["winner_rank"]) - np.log1p(df["loser_rank"]) 
    df_winner["age_diff"] = df["winner_age"] - df["loser_age"]
    df_winner["label"] = 1
    df_winner["same_country"] = (df["winner_ioc"] == df["loser_ioc"]).astype(int)
    df_winner["home_advantage"] = (df["winner_ioc"] == df["tourney_id"].str[:3]).astype(int)
    
    df_loser = df.copy()
    df_loser["rank_diff"] = -df_winner["rank_diff"]
    df_loser["age_diff"] = -df_winner["age_diff"]
    df_loser["label"] = 0
    df_loser["same_country"] = df_winner["same_country"]
    df_loser["home_advantage"] = (df["loser_ioc"] == df["tourney_id"].str[:3]).astype(int)
    
    df_final = pd.concat([df_winner, df_loser], ignore_index=True)
    
    df_final = calculate_elo_features(df_final)
    df_final = add_head_to_head_feature(df_final)
    df_final = add_recent_winrate(df_final)
    
    surfaces = ['Hard', 'Clay', 'Grass']
    for surface in surfaces:
        df_final[f"surface_{surface.lower()}"] = (df_final["surface"] == surface).astype(int)
    
    return df_final.drop(columns=["surface", "tourney_id"], errors="ignore")