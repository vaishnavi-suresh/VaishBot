from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from Foundation import NSData,  NSUnarchiver
from sentence_transformers import SentenceTransformer
import os
import sqlite3
import torch
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


    

def decode(data: bytes):
    if not data:
        return None

    nsdata = NSData.dataWithBytes_length_(data, len(data))

    try:
        obj = NSUnarchiver.unarchiveObjectWithData_(nsdata)
        if obj is not None and hasattr(obj, "string"):
            return str(obj.string())
    except Exception:
        pass






def seed():
    #connect the database here
    client = MongoClient(os.getenv("URI_LINK"), server_api=ServerApi('1'))
    coll = client['rag']['messages']
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    #set up model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    model = SentenceTransformer("nomic-ai/modernbert-embed-base", device = device)

    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "chat.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT text, attributedBody, handle_id, associated_message_type FROM message")
    rows = cursor.fetchall()

    results = []
    for text, attr, handleid, atp in rows:

        if handleid != 0 or atp == 2000:
            continue
        if text:
    
            results.append(text)
        elif attr:
            decoded = decode(attr)
            if decoded:
                results.append(decoded)
            else:
                continue
        else:
            continue

    results = results[41617:100000]
    embeds = model.encode(results,normalize_embeddings=True, batch_size=32, show_progress_bar=True).tolist()
    for i, e in enumerate(embeds):
        try:
            coll.insert_one(
                {
                    "text": results[i],
                    "embedding": e
                }
            ) 
        except:
            continue
    conn.close()

    


def main():
    seed()



if __name__ == "__main__":
    main()
