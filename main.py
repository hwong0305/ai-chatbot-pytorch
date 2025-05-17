from random import random
from fastapi import FastAPI
from pydantic import BaseModel
from bot import ChatbotAssistant

def get_stocks(quantity=3):
  stocks = ["APPL", "META", "NVDA", "GS", "MSFT"]
  return random.sample(stocks, quantity)

class ChatMessage(BaseModel):
  message: str

app = FastAPI()
assistant = ChatbotAssistant("intents.json", function_mappings={"stocks": get_stocks, "get_stocks": get_stocks})
assistant.parse_intents()
assistant.load_model("chatbot_model.pth", "dimensions.json")

@app.get('/')
async def root():
  return "Hello World"

@app.post('/chat')
async def chat(message: ChatMessage):
  return assistant.process_message(message.message)