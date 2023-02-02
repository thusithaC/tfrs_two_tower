
from pymongo import MongoClient

HOST = "localhost"
PORT = 27017


def insert_one(document, db="modelExperiments", collection='recomendation1'):
    client = MongoClient(host=HOST, port=PORT)
    db_con = client.get_database(db)
    collection = db_con.get_collection(collection)
    result = collection.insert_one(document)
    print('Created {0}'.format(result.inserted_id))
    client.close()
    return

