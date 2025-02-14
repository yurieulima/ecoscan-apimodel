from fastapi import FastAPI
from routes.image_router import router as image_router

app = FastAPI()

app.include_router(image_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

