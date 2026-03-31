from fastapi import FastAPI
from random import choice

app = FastAPI()


JOKES = [
    "joke_1",
    "joke_2",
    "joke_3",
    "joke_4",
    "joke_5",
    "joke_6",
    "joke_7",
    "joke_8",
    "joke_9"
]

@app.get("/joke")
def joke():
    return {"joke" : choice(JOKES)}
