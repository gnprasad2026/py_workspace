from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from groq import Groq
import os

load_dotenv()

app = FastAPI(title="GROQ JOKE API")

def generate_joke(topic:str) -> str:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing api_key in env file")
    else:
        print(f"GROQ_API_KEY : {GROQ_API_KEY}")

    GROQ_MODEL = os.getenv("GROQ_MODEL")    
    print(f"GROQ_MODEL : {GROQ_MODEL}")

    client = Groq()
    
    completions = client.chat.completions.create(
        model = GROQ_MODEL,
        messages=[
            {"role":"system", "content": "You are a concise joke writer."},
            {"role":"user", "content": f"write one short, funny joke about the given {topic}."}
        ],
        temperature=0.8,
        max_tokens=100,
    )

    joke = completions.choices[0].message.content.strip()
    if not joke:
        raise HTTPException(status_code=500, detail="LLM returnd an empty joke")
    
    return joke.strip()

@app.get("/joke")
def get_joke_from_llm_model(topic:str=Query(..., min_length=1, description="Topic for the joke")) -> dict[str, str]:
    return {"topic": topic,"Joke ": generate_joke(topic)}




