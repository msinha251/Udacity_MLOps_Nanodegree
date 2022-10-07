from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    item: str

app = FastAPI()

@app.get("/items")
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]

@app.post("/{item_id}")
def create_item(item_id: int, q: str, item: Item):
    return {"item_id": item_id, "q": q, "item": item}