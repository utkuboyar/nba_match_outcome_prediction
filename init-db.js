db = db.getSiblingDB("nba_games_db");
db.games.drop();

db.games.insertMany([
    {
        "id": 1,
        "home_team": "Golden State Warriors",
        "away_team": "Los Angeles Lakers",
        "predicted_proba_home": "0.65",
        "predicted_proba_away": "0.35",
        "predicted_odds_home": "1.538",
        "predicted_odds_away": "2.857",
        "status":"correct",
        "score": "122:109"
    },
    {
        "id": 2,
        "home_team": "Atlanta Hawks",
        "away_team": "Boston Celtics",
        "predicted_proba_home": "0.40",
        "predicted_proba_away": "0.60",
        "predicted_odds_home": "2.500",
        "predicted_odds_away": "1.667",
        "status":"wrong",
        "score": "110:105"
    },
    {
        "id": 3,
        "home_team": "Cleveland Cavaliers",
        "away_team": "Chicago Bulls",
        "predicted_proba_home": "0.57",
        "predicted_proba_away": "0.43",
        "predicted_odds_home": "1.754",
        "predicted_odds_away": "2.326",
        "status":"not played",
        "score": null
    },
]);