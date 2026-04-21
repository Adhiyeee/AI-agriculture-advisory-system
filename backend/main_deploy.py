from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ml_model

app = FastAPI(title="Agri Deploy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
@app.on_event("startup")
def load():
    ml_model.load_or_train()

@app.get("/")
def home():
    return {"status": "running"}

# SAME ROUTE AS YOUR ORIGINAL BACKEND ✅
@app.post("/analysis/run")
def run_analysis(body: dict):

    crop = body.get("crop", "")
    state = body.get("state", "")
    district = body.get("district", "")

    # Get state-based inputs
    resolved = ml_model.resolve_state_inputs(state, {})
    inputs = {
        "N": resolved["N"],
        "P": resolved["P"],
        "K": resolved["K"],
        "temperature": resolved["temperature"],
        "humidity": resolved["humidity"],
        "ph": resolved["ph"],
        "rainfall": resolved["rainfall"],
    }

    # ML prediction
    ml = ml_model.compute_suitability(
        crop_name=crop,
        N=inputs["N"],
        P=inputs["P"],
        K=inputs["K"],
        temperature=inputs["temperature"],
        humidity=inputs["humidity"],
        ph=inputs["ph"],
        rainfall_annual=inputs["rainfall"],
        state_major_crops=""
    )

    return {
        "success": True,
        "data": {
            "suitability_score": ml["suitability_score"],
            "rating": ml["rating"],
            "rf_probability": ml["rf_probability"],
            "crop": crop,
            "state": state,
            "district": district
        }
    }