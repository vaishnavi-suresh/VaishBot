from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from Foundation import NSData,  NSUnarchiver
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import torch
import json
from dotenv import load_dotenv

#Global information
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
model_name = "Qwen/Qwen3-30B-A3B"



def _shrink_context(ctx: str, max_chars: int = 12000) -> str:  # CHANGED
    return ctx if len(ctx) <= max_chars else ctx[: max_chars]

def topk(query,k):
    #set up the client
    client = MongoClient(os.getenv("URI_LINK"), server_api=ServerApi('1'))
    coll = client['rag']['messages']

    #set up the model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("nomic-ai/modernbert-embed-base", device = device)

    qembeds = model.encode([query],normalize_embeddings=True, batch_size=32, show_progress_bar=False)[0].tolist()
    num_candidates = max(2000, k*15)
    num_candidates = min(10000, num_candidates)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",   
                "path": "embedding",      
                "queryVector": qembeds,
                "numCandidates": num_candidates,
                "limit": k
            }
        },
        {   # include the score
            "$project": {
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(coll.aggregate(pipeline))



def generate_messages():
    query = "what is SVS"

    l = topk(query,1000)



    context= ""
    for doc in l:  
        context += f'\n{json.dumps(doc, indent=2, default=str)}'
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_KEY'),
    
    )
    context = _shrink_context(context, max_chars=12000)

    prompt = f"""Query: {query}\n\n Context: Your name is Vaish (short for Vaishnavi). These are some of the texts you have sent: {context}. When responding to the prompt, be sure to use the same tone in the messages. 
            try to match the tone of these texts as if you are this person. Pretend to be this person.
            remember not to overfit the context,DO NOT MAKE THINGS UP, SAY NOTHING YOU ARENT SURE OF.
            make sure the responses make sense"""
    c = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    temperature=0.05,
    top_p=0.95,
    messages = [{
        "role": "user",
        "content": prompt,
        
    }]
    )
    print(c.choices[0].message.content)




def main():
    
    generate_messages()



if __name__ == "__main__":
    main()


