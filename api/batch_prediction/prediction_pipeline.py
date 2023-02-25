from .data_processing.post_game_workflow import PostGameWorkflow
from .data_processing.pre_game_workflow import PreGameWorkflow
from .database_control import DatabaseController
from .predictor import Predictor


import _pickle
import datetime

from time import time

class BatchPredictionPipeline:
    @staticmethod
    def run():
        db_controller = DatabaseController()
        
        post_game = PostGameWorkflow(db_controller=db_controller)
        post_game.update_recent_dfs()

        pre_game = PreGameWorkflow()
        new_games = pre_game.prepare_new_games()

        if new_games is not None:
            predictor = Predictor(new_games)
            predictions = predictor.get_predictions()
            
            # db_controller.add_predictions(predictions)
            db_controller.add_predictions(predictions, new_games[['GAME_ID', 'home_odds', 'away_odds']].set_index('GAME_ID')) # new
