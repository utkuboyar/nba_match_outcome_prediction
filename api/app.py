from flask import Flask, jsonify, request, redirect, url_for
# import pymongo 
from pymongo import MongoClient
from bson.objectid import ObjectId
from batch_prediction.performance_tracking import Monitor

from batch_prediction.prediction_pipeline import BatchPredictionPipeline
from batch_prediction.utils.db_connection import get_collection
from betting_optimization.multiple_initializations import BettingOptimizer
from time import time

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify('hello world')

@app.route('/games', methods=['GET'])
def get_games():
    games_collection = get_collection()

    all_games = []
    for game in games_collection.find({}, {"_id":0, "game_id": 1, "game_date": 1, "home_team": 1, "away_team": 1, "predicted_proba_home":1,
                                           "predicted_proba_away":1, "predicted_odds_home":1, "predicted_odds_away":1,
                                           "predicted_label": 1, "status":1, "score":1}):
        all_games.append(game)

    if all_games == []:
        return 'none'
    
    return jsonify(all_games)

@app.route('/add_game', methods=['POST'])
def add_game():
    games_collection = get_collection()

    home_team, away_team = request.json['home_team'], request.json['away_team']
    predicted_proba_home, predicted_proba_away = request.json['predicted_proba_home'], request.json['predicted_proba_away']
    predicted_odds_home, predicted_odds_away = request.json['predicted_odds_home'], request.json['predicted_odds_away']
    status, score = request.json['status'], request.json['score']

    games_collection.insert_one({"home_team": home_team, "away_team": away_team, "predicted_proba_home":predicted_proba_home,
                            "predicted_proba_away":predicted_proba_away, "predicted_odds_home":predicted_odds_home, 
                            "predicted_odds_away":predicted_odds_away, "status":status, "score":score})

    return 'ok'

@app.route('/update_game', methods=['POST'])
def update_game():
    games_collection = get_collection()

    game_id = request.json['game_id']
    updated_game = request.json['updated_game']
    games_collection.update_one({'_id':ObjectId(game_id)}, {"$set": updated_game})
    return redirect(url_for('get_games'))


@app.route('/calculate_bet_ratios', methods=['POST'])
def calculate_bet_ratios():
    game_ids = request.json['game_ids']
    bet_for = request.json['bet_for']
    all_odds = request.json['all_odds']
    is_api_endpoint = request.json['is_api_endpoint']

    games_collection = get_collection('games')
    probs, odds = [], []
    for game_id, supported_team, two_odds in zip(game_ids, bet_for, all_odds):
        game = games_collection.find_one({'game_id':game_id}, {'predicted_proba_home':1, 'predicted_proba_away':1})
        if supported_team == 'home':
            probs.append(float(game['predicted_proba_home']))
            odds.append(float(two_odds['home']))
        elif supported_team == 'away':
            probs.append(float(game['predicted_proba_away']))
            odds.append(float(two_odds['away']))
        else:
            raise Exception('you should bet for one of home and away team')

    optimizer = BettingOptimizer(probs=probs, odds=odds)
    result = optimizer.run()
    if result == 'all expected profits are negative':
        return result
    else:
        if is_api_endpoint:
            print('here')
            res = tuple(str(ratio) for ratio in result[0])
            print(res)
            return res
        else:
            return f'The best betting strategy is to place bets on the games with the ratios {result[0]}, respectively. Sharpe ratio: {result[1]}'
        
@app.route('/batch_results', methods=['GET'])
def get_batch_results():
    batch_date = request.json['batch_date']
    monitor = Monitor()
    results = monitor.get_batch_results(batch_date)
    return jsonify(results)

@app.route('/interval_results', methods=['GET'])
def get_interval_results():
    start_date = request.json['start_date']
    end_date = request.json['end_date']
    monitor = Monitor()
    results = monitor.get_performance_on_time_interval(start_date, end_date)
    return jsonify(results)

@app.route('/prediction_pipeline', methods=['GET'])
def update_predict():
    t0 = time()
    BatchPredictionPipeline.run()
    return str(time() - t0)

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
