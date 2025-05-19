import os
from fastapi import FastAPI, requests
import uvicorn
from pydantic import BaseModel
from typing import Any
from Model import Model
from VectorStoreManager import VectorStoreManager

class Message(BaseModel):
    role: str
    content: str


inferenceServer = FastAPI()


base_dir = os.path.dirname(os.path.abspath(__file__))
faiss_path = os.path.join(base_dir, "..", "Data", "SlicerFAISS")
manager = VectorStoreManager(faiss_path)
chatbot = Model(manager=manager)
print("okey")

@inferenceServer.post("/generate")
async def generate(message: Message):
    print(message)
    response = chatbot.generate_response(message.content)
    print(response)
    return response
    
if __name__=="__main__":
    uvicorn.run(inferenceServer, host="127.0.0.1", port=81, log_level="info")