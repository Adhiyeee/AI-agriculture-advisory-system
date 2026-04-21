from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ml_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load():
    ml_model.load_or_train()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    try:
        result = ml_model.compute_suitability(
            crop_name=data["crop"],
            N=data["N"],
            P=data["P"],
            K=data["K"],
            temperature=data["temperature"],
            humidity=data["humidity"],
            ph=data["ph"],
            rainfall_annual=data["rainfall"],
            state_major_crops=""
        )

        return {
            "score": result["suitability_score"],
            "rating": result["rating"]
        }

    except Exception as e:
        return {"error": str(e)}