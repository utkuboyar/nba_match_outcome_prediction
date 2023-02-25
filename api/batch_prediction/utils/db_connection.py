from pymongo import MongoClient

def get_collection(collection_name='games'):
    client = MongoClient(host='localhost',
                         port=27019, 
                         username='root', 
                         password='123456dev',
                         authSource="admin")
    db = client["nba_games_db"]
    games_collection = db[collection_name]
    return games_collection