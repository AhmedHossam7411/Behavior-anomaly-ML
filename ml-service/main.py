from fastapi import FastAPI
from routes.predict import router

app = FastAPI()

@app.on_event("startup")
def load_model():
    from models.tabpfn_model import MLModel
    app.state.model = MLModel()  
app.include_router(router)