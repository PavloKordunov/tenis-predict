from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def train_and_evaluate(df):
    X = df[["rank_diff", "age_diff", "surface_hard", "surface_clay", "surface_grass",
        "elo_diff", "h2h_winrate", "recent_winrate_diff", "same_country", "home_advantage"]]
    y = df["label"]
    
    
    clf1 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty='l2',
            C=0.5,
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
    )
    clf2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        max_features='sqrt',
        random_state=42
    )
    clf3 = XGBClassifier(
        eval_metric='logloss',
        learning_rate=0.01,
        max_depth=6,
        n_estimators=500,
        subsample=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0
    )

    model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgp', clf3)], voting='soft')

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv)
    print(f"Accuracy: {scores.mean():.2%} ± {scores.std():.2%}")

    model.fit(X, y)
    
    return model

