"""
Agriculture Advisory System – FastAPI Backend v4
=================================================
KEY CHANGES:
  - Soil/climate data now READ FROM india_state_climate.csv (not hardcoded)
  - ml_model.resolve_state_inputs() does the CSV lookup
  - RF model now covers 32 crops (added wheat, mustard, potato, etc.)
  - Advisory text references the actual score percentage meaningfully
  - /ml/info shows full accuracy report
  - /ml/state/{state} shows what CSV data we have for that state
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from bson import ObjectId
import random, string, os, httpx, json
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jose import JWTError, jwt
from dotenv import load_dotenv

import ml_model

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URL      = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME        = os.getenv("DB_NAME", "agriculture_advisory")
JWT_SECRET     = os.getenv("JWT_SECRET", "change-this-secret-in-production")
JWT_ALGORITHM  = "HS256"
JWT_EXPIRE_MIN = 60 * 24
OTP_EXPIRE_MIN = 10
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GMAIL_USER     = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASS = os.getenv("GMAIL_APP_PASS", "")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Agriculture Advisory System")

app = FastAPI(title="Agriculture Advisory System API", version="4.0.0-RF")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

mongo_client = None
db = None

@app.on_event("startup")
async def startup():
    global mongo_client, db
    ml_model.load_or_train()
    mongo_client = AsyncIOMotorClient(MONGO_URL)
    db = mongo_client[DB_NAME]
    await db.users.create_index("email", unique=True)
    await db.otps.create_index("email")
    await db.otps.create_index("expires_at", expireAfterSeconds=0)
    await db.analyses.create_index("user_email")
    await db.analyses.create_index([("user_email", 1), ("created_at", -1)])
    print(f"✅ MongoDB connected | Agriculture Advisory v4.0-RF")
    print(f"📧 Gmail: {'✓ ' + GMAIL_USER if GMAIL_USER and GMAIL_APP_PASS else '✗ DEV mode'}")
    print(f"🤖 Claude: {'✓ configured' if ANTHROPIC_KEY else '✗ fallback'}")

@app.on_event("shutdown")
async def shutdown():
    mongo_client.close()

# ── Gmail SMTP ────────────────────────────────────────────────────────────────
def send_otp_gmail(to_email: str, to_name: str, otp: str) -> bool:
    if not GMAIL_USER or not GMAIL_APP_PASS:
        print(f"[DEV] OTP for {to_email}: {otp}")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your OTP: {otp}"
        msg["From"]    = f"{SMTP_FROM_NAME} <{GMAIL_USER}>"
        msg["To"]      = to_email
        html = f"""<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:40px 20px;">
<div style="max-width:500px;margin:auto;background:white;border-radius:16px;overflow:hidden;">
<div style="background:linear-gradient(135deg,#2D6A4F,#52B788);padding:30px;text-align:center;">
<h1 style="color:white;margin:0;">🌾 Agriculture Advisory</h1></div>
<div style="padding:32px;">
<h2 style="color:#2C1A0E;">Hello, {to_name}!</h2>
<p style="color:#6B7280;">Your one-time password:</p>
<div style="background:#E8F5E9;border:2px dashed #2D6A4F;border-radius:12px;padding:24px;text-align:center;margin:20px 0;">
<div style="font-size:36px;font-weight:900;letter-spacing:0.3em;color:#2C1A0E;font-family:monospace;">{otp}</div>
<div style="color:#6B7280;font-size:12px;margin-top:8px;">Valid for {OTP_EXPIRE_MIN} minutes only</div>
</div>
<p style="color:#6B7280;font-size:13px;">Never share this code with anyone.</p>
</div></div></body></html>"""
        msg.attach(MIMEText(html, "html"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo(); s.starttls(context=ctx); s.ehlo()
            s.login(GMAIL_USER, GMAIL_APP_PASS)
            s.sendmail(GMAIL_USER, to_email, msg.as_string())
        print(f"📧 OTP sent → {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email error: {e} | [DEV] OTP for {to_email}: {otp}")
        return False

# ── Auth helpers ──────────────────────────────────────────────────────────────
def make_otp(n=6):  return ''.join(random.choices(string.digits, k=n))
def make_jwt(email):
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN)
    return jwt.encode({"sub": email, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGORITHM)
def check_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])["sub"]
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

security = HTTPBearer()
async def current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    email = check_jwt(creds.credentials)
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user: raise HTTPException(401, "User not found")
    return user

# ── Schemas ───────────────────────────────────────────────────────────────────
class OTPRequest(BaseModel):
    email: EmailStr
    name:  str = "Farmer"

class VerifyRequest(BaseModel):
    email: EmailStr
    otp:   str

class AnalysisRequest(BaseModel):
    # New format (v5): state + district separate
    state:       str   = ""
    district:    str   = ""
    crop:        str
    season:      str   = ""
    # Old format (v3/v4) backward compat: "location" field
    location:    str   = ""
    # Optional soil/climate overrides (override CSV values)
    N:           float = None
    P:           float = None
    K:           float = None
    temperature: float = None
    humidity:    float = None
    ph:          float = None
    rainfall:    float = None   # annual mm

class RecommendRequest(BaseModel):
    state:       str
    district:    str   = ""
    season:      str   = ""
    N:           float
    P:           float
    K:           float
    temperature: float
    humidity:    float
    ph:          float
    rainfall:    float  # annual mm

class ContactRequest(BaseModel):
    name: str; email: EmailStr; subject: str = ""; message: str

# ── Advisory narrative with Claude (or fallback) ──────────────────────────────
async def generate_narrative(score, rating, crop, state, district,
                              season, rf_prob, inputs, state_data) -> dict:
    location = f"{district}, {state}" if district else state

    # Build score interpretation line
    if score >= 80:
        score_interp = f"strong suitability — the RF model is {rf_prob:.0f}% confident"
    elif score >= 65:
        score_interp = f"good suitability with minor limitations — RF confidence {rf_prob:.0f}%"
    elif score >= 50:
        score_interp = (f"moderate suitability — some factors fall short "
                        f"(RF confidence only {rf_prob:.0f}%)")
    elif score >= 35:
        score_interp = (f"poor suitability — significant mismatches found "
                        f"(RF confidence {rf_prob:.0f}%)")
    else:
        score_interp = (f"not recommended — major incompatibilities "
                        f"(RF confidence {rf_prob:.0f}%)")

    # What's limiting the score
    ideal = ml_model.CROP_IDEAL.get(crop.lower(), {})
    limiting = []
    if ideal:
        r_lo, r_hi = ideal.get("rainfall", (0, 9999))
        t_lo, t_hi = ideal.get("temperature", (0, 99))
        if not (r_lo <= inputs["rainfall"] <= r_hi):
            diff = "low" if inputs["rainfall"] < r_lo else "high"
            limiting.append(f"rainfall is {diff} ({inputs['rainfall']:.0f}mm vs ideal {r_lo}–{r_hi}mm)")
        if not (t_lo <= inputs["temperature"] <= t_hi):
            diff = "cold" if inputs["temperature"] < t_lo else "hot"
            limiting.append(f"temperature is too {diff} ({inputs['temperature']:.1f}°C vs ideal {t_lo}–{t_hi}°C)")
        n_lo, n_hi = ideal.get("N", (0, 999))
        if not (n_lo <= inputs["N"] <= n_hi):
            limiting.append(f"nitrogen is {'low' if inputs['N'] < n_lo else 'high'} ({inputs['N']:.0f} kg/ha vs ideal {n_lo}–{n_hi})")

    limiting_str = "; ".join(limiting) if limiting else "all key parameters are within acceptable range"

    climate_zone = state_data.get("climate_zone", "Tropical") if state_data else "Tropical"
    soil_type    = state_data.get("soil_type", "Mixed") if state_data else "Mixed"
    major_crops  = state_data.get("major_crops", "") if state_data else ""

    prompt = f"""You are a senior Indian agriculture scientist writing an advisory report for a farmer.

CROP REQUESTED: {crop}
LOCATION: {location}
CLIMATE ZONE: {climate_zone}
DOMINANT SOIL: {soil_type}
SEASON: {season or "Not specified"}
MAJOR CROPS IN THIS STATE: {major_crops}

MODEL RESULTS:
  Suitability Score: {score}/100 → {score_interp}
  Rating: {rating}
  Random Forest Confidence: {rf_prob:.1f}%

ACTUAL SOIL & CLIMATE DATA (from india_state_climate.csv):
  Nitrogen (N):   {inputs['N']:.0f} kg/ha
  Phosphorus (P): {inputs['P']:.0f} kg/ha  
  Potassium (K):  {inputs['K']:.0f} kg/ha
  Soil pH:        {inputs['ph']:.1f}
  Avg Temperature:{inputs['temperature']:.1f}°C
  Avg Humidity:   {inputs['humidity']:.0f}%
  Annual Rainfall:{inputs['rainfall']:.0f} mm

WHAT IS LIMITING THE SCORE: {limiting_str}

Write ONLY valid JSON (no markdown fences) with these exact keys:
{{
  "verdict_title": "One punchy verdict under 12 words that mentions the score {score}/100",
  "verdict_text": "3 sentences: (1) state what the {score}/100 means for this specific crop here, (2) name the specific limiting factor or strength from the data above, (3) give one immediate actionable recommendation based on the score.",
  "rainfall_advice": "One sentence: is {inputs['rainfall']:.0f}mm adequate for {crop}? Mention the ideal range and what to do if short/excess.",
  "temperature_advice": "One sentence about whether {inputs['temperature']:.1f}°C suits {crop} and any management tip.",
  "soil_advice": "One sentence about N:{inputs['N']:.0f}/P:{inputs['P']:.0f}/K:{inputs['K']:.0f} and pH:{inputs['ph']:.1f} for {crop} — are they adequate or what to add?",
  "detailed_advisory": "Write exactly 4 paragraphs separated by \\n\\n. Each must be 3-5 sentences and directly reference the {score}/100 score and actual numbers above:\\n  Para 1 — Score interpretation: explain precisely why the score is {score} — which factors contributed most, which dragged it down. Use the actual numbers.\\n  Para 2 — Cultivation guide tailored to {score}/100: if >75 give specific variety, sowing dates, spacing for {location}. If 50-75 explain what management compensates for the gap. If <50 explain what would need to change to make it viable.\\n  Para 3 — Risk management scaled to {score}/100: if score is high warn about peak risks. If moderate/low explain why those risks are amplified by the limiting factors.\\n  Para 4 — Market and economics: yield expectation at this score level, current MSP or market rate, profit viability at {score}/100."
}}"""

    if not ANTHROPIC_KEY:
        return _fallback(crop, location, score, rating, rf_prob, inputs, limiting_str)

    try:
        async with httpx.AsyncClient(timeout=30) as h:
            r = await h.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1400,
                      "messages": [{"role": "user", "content": prompt}]},
            )
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        clean = text.strip()
        # Strip markdown fences if present
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception as e:
        print(f"Claude error: {e}")
        return _fallback(crop, location, score, rating, rf_prob, inputs, limiting_str)


def _fallback(crop, location, score, rating, rf_prob, inputs, limiting_str) -> dict:
    """Deterministic fallback that still references actual numbers."""
    ideal = ml_model.CROP_IDEAL.get(crop.lower(), {})
    rain_lo, rain_hi = ideal.get("rainfall", (500, 1500))
    temp_lo, temp_hi = ideal.get("temperature", (15, 35))

    if score >= 80:
        vt = f"{crop} is excellent for {location} — score {score}/100"
        vv = (f"The RF model gives {crop} a score of {score}/100 in {location}, "
              f"with {rf_prob:.0f}% Random Forest confidence. "
              f"Your state's soil (N:{inputs['N']:.0f}, P:{inputs['P']:.0f}, K:{inputs['K']:.0f} kg/ha) "
              f"and climate ({inputs['temperature']:.1f}°C, {inputs['rainfall']:.0f}mm rain) "
              f"closely match {crop}'s ideal requirements. "
              f"You can expect strong yields with standard agronomic practices.")
        adv = (
            f"At {score}/100, {crop} is a highly suitable choice for {location}. "
            f"The Random Forest model, trained on 3,700+ samples, assigns {rf_prob:.0f}% confidence to this crop here. "
            f"Your state's annual rainfall of {inputs['rainfall']:.0f}mm is {'within' if rain_lo <= inputs['rainfall'] <= rain_hi else 'close to'} "
            f"the ideal range of {rain_lo}–{rain_hi}mm for {crop}.\n\n"
            f"Select high-yielding certified varieties from your state agricultural university. "
            f"For {location} with temperatures averaging {inputs['temperature']:.1f}°C, "
            f"sow during the appropriate season and maintain recommended plant spacing. "
            f"Apply fertilisers based on the ideal N/P/K balance for {crop}.\n\n"
            f"At this high score, the main risks are complacency — ensure timely irrigation, "
            f"pest monitoring, and proper post-harvest handling. "
            f"Soil pH of {inputs['ph']:.1f} should be {'maintained' if 5.5 <= inputs['ph'] <= 7.5 else 'corrected'} "
            f"in the optimal range for {crop}.\n\n"
            f"With a score of {score}/100, you can expect yields near the upper range for this crop. "
            f"Check agmarknet.gov.in for current market rates and compare with the applicable MSP "
            f"before deciding your selling channel."
        )
    elif score >= 50:
        vt = f"{crop} is viable in {location} — score {score}/100, manage {limiting_str.split(';')[0] if limiting_str else 'inputs'}"
        vv = (f"The RF model scores {crop} at {score}/100 in {location} ({rf_prob:.0f}% RF confidence). "
              f"This moderate score is primarily because {limiting_str}. "
              f"With corrective management — especially irrigation or soil amendments — yields can be improved.")
        adv = (
            f"At {score}/100, {crop} is viable but faces some limitations in {location}. "
            f"The key limiting factor is: {limiting_str}. "
            f"State rainfall of {inputs['rainfall']:.0f}mm compares to the ideal {rain_lo}–{rain_hi}mm for {crop}.\n\n"
            f"Choose drought-tolerant or stress-resistant varieties if rainfall is limiting. "
            f"Supplement with irrigation during critical growth stages. "
            f"Apply {crop}-specific fertiliser schedule to compensate for any NPK gaps.\n\n"
            f"At a moderate score, {crop} is more susceptible to weather shocks. "
            f"Maintain crop insurance and monitor weather forecasts closely. "
            f"Apply preventive fungicide/pesticide treatments at key growth stages.\n\n"
            f"Expect yields at 60–75% of the optimal range at this score level. "
            f"Focus on quality over quantity and sell through APMC or FPOs for better pricing. "
            f"The lower yield means higher per-unit cost — ensure market price covers expenses."
        )
    else:
        vt = f"{crop} has poor suitability for {location} — score {score}/100"
        vv = (f"The RF model assigns only {score}/100 to {crop} in {location} "
              f"({rf_prob:.0f}% RF confidence). "
              f"The main issues: {limiting_str}. "
              f"Consider the alternative crops shown below for better results.")
        adv = (
            f"At {score}/100, {crop} faces significant challenges in {location}. "
            f"The Random Forest model, with {rf_prob:.0f}% confidence, indicates this crop is poorly matched. "
            f"Critical mismatches: {limiting_str}.\n\n"
            f"If you still wish to grow {crop}, major interventions are needed: "
            f"extensive irrigation infrastructure, greenhouse or protected cultivation, "
            f"and soil amendment to match the {crop} requirements. "
            f"These add significantly to cost and risk.\n\n"
            f"At this score, {crop} is highly vulnerable to pests, diseases, and weather stress "
            f"because the crop will already be under environmental stress. "
            f"Crop failure risk is elevated — do not invest heavily without expert consultation from your local KVK.\n\n"
            f"Economically, below-optimal yields combined with high management costs make {crop} "
            f"difficult to justify in {location} at this score. "
            f"Review the alternative crop recommendations — they offer much better returns "
            f"for your soil and climate."
        )

    return {
        "verdict_title": vt,
        "verdict_text":  vv,
        "rainfall_advice": (
            f"Your state receives {inputs['rainfall']:.0f}mm annually; {crop} ideally needs "
            f"{rain_lo}–{rain_hi}mm — {'sufficient' if rain_lo <= inputs['rainfall'] <= rain_hi else 'supplement with irrigation'}."
        ),
        "temperature_advice": (
            f"Average temperature of {inputs['temperature']:.1f}°C is "
            f"{'ideal' if temp_lo <= inputs['temperature'] <= temp_hi else 'outside the optimal'} "
            f"range of {temp_lo}–{temp_hi}°C for {crop}."
        ),
        "soil_advice": (
            f"Soil N:{inputs['N']:.0f}, P:{inputs['P']:.0f}, K:{inputs['K']:.0f} kg/ha at pH {inputs['ph']:.1f} — "
            f"apply targeted fertilisers to reach {crop}'s ideal NPK balance."
        ),
        "detailed_advisory": adv,
    }

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    info = ml_model.get_model_info()
    return {
        "status": "ok", "version": "4.0.0-RF",
        "ml_engine": "Random Forest",
        "rf_test_accuracy":  info.get("test_accuracy_pct", "?"),
        "rf_cv_accuracy":    info.get("cv_accuracy_pct", "?"),
        "rf_crops":          info.get("n_crops", "?"),
        "climate_csv_loaded": ml_model._climate_df is not None,
        "gmail_configured":  bool(GMAIL_USER and GMAIL_APP_PASS),
        "claude_configured": bool(ANTHROPIC_KEY),
    }

@app.get("/health")
async def health():
    try:
        await mongo_client.admin.command("ping")
        return {"status": "ok", "database": "connected", "ml": "random_forest_active"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/auth/send-otp")
async def send_otp(body: OTPRequest, background_tasks: BackgroundTasks):
    otp = make_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_EXPIRE_MIN)
    await db.otps.update_one({"email": body.email},
        {"$set": {"otp": otp, "expires_at": expires_at, "name": body.name}}, upsert=True)
    await db.users.update_one({"email": body.email},
        {"$setOnInsert": {"email": body.email, "name": body.name,
                          "created_at": datetime.utcnow()}}, upsert=True)
    background_tasks.add_task(send_otp_gmail, body.email, body.name, otp)
    gmail_ready = bool(GMAIL_USER and GMAIL_APP_PASS)
    resp = {"success": True, "expires_in_minutes": OTP_EXPIRE_MIN,
            "email_configured": gmail_ready}
    if gmail_ready:
        resp["message"]  = f"OTP sent to {body.email}"
        resp["greeting"] = f"Hi {body.name}! Check your inbox."
    else:
        resp["otp"]      = otp
        resp["dev_mode"] = True
        resp["message"]  = "Dev mode — add GMAIL_USER + GMAIL_APP_PASS to .env for real email"
        resp["greeting"] = f"Hi {body.name}! (Dev mode)"
    return resp

@app.post("/auth/verify-otp")
async def verify_otp(body: VerifyRequest):
    rec = await db.otps.find_one({"email": body.email})
    if not rec: raise HTTPException(400, "No OTP found.")
    if datetime.utcnow() > rec["expires_at"]:
        await db.otps.delete_one({"email": body.email})
        raise HTTPException(400, "OTP expired.")
    if rec["otp"] != body.otp.strip(): raise HTTPException(400, "Incorrect OTP.")
    await db.otps.delete_one({"email": body.email})
    await db.users.update_one({"email": body.email},
        {"$set": {"last_login": datetime.utcnow(), "name": rec.get("name", "Farmer")}})
    user = await db.users.find_one({"email": body.email}, {"_id": 0})
    return {"success": True, "token": make_jwt(body.email), "user": user}

@app.get("/auth/me")
async def get_me(user=Depends(current_user)): return user

# ── Analysis Run — USES CSV DATA ──────────────────────────────────────────────
@app.post("/analysis/run")
async def run_analysis(body: AnalysisRequest, user=Depends(current_user)):
    # Resolve state: accept both new format (state field) and old format (location field)
    raw_state    = (body.state or "").strip()
    raw_location = (body.location or "").strip()
    district_raw = (body.district or "").strip()
    crop_raw     = body.crop.strip()
    season       = (body.season or "").strip()

    # If state is empty but location is "District, State" or just "State", parse it
    if not raw_state and raw_location:
        if "," in raw_location:
            parts     = raw_location.split(",", 1)
            district_raw = district_raw or parts[0].strip()
            raw_state = parts[1].strip()
        else:
            raw_state = raw_location

    state_raw = raw_state or raw_location  # final fallback

    # Debug log (remove in production)
    print(f"[API] state='{state_raw}' district='{district_raw}' crop='{crop_raw}' season='{season}'")

    # 1. Resolve inputs from CSV + any user overrides
    overrides = {k: getattr(body, k) for k in ["N","P","K","temperature","humidity","ph","rainfall"]}
    resolved  = ml_model.resolve_state_inputs(state_raw, overrides)
    print(f"[API] Data source: {resolved['source']} | rain={resolved['rainfall']}mm | temp={resolved['temperature']}C")
    inputs    = {k: resolved[k] for k in ["N","P","K","temperature","humidity","ph","rainfall"]}
    state_data = resolved.get("state_data")
    data_source = resolved.get("source", "unknown")

    # 2. RF suitability score
    major_crops = state_data.get("major_crops", "") if state_data else ""
    ml = ml_model.compute_suitability(
        crop_name    = crop_raw,
        N            = inputs["N"],
        P            = inputs["P"],
        K            = inputs["K"],
        temperature  = inputs["temperature"],
        humidity     = inputs["humidity"],
        ph           = inputs["ph"],
        rainfall_annual     = inputs["rainfall"],
        state_major_crops   = major_crops,
    )
    score   = ml["suitability_score"]
    rating  = ml["rating"]
    rf_prob = ml["rf_probability"]

    # 3. Alternative crops
    alternatives = ml_model.get_top_alternatives(
        exclude_crop      = crop_raw,
        N                 = inputs["N"],
        P                 = inputs["P"],
        K                 = inputs["K"],
        temperature       = inputs["temperature"],
        humidity          = inputs["humidity"],
        ph                = inputs["ph"],
        rainfall_annual   = inputs["rainfall"],
        top_n             = 3,
        state_major_crops = major_crops,
    )

    # 4. Generate advisory text
    narrative = await generate_narrative(
        score, rating, crop_raw, state_raw, district_raw,
        season, rf_prob, inputs, state_data)

    location_display = f"{district_raw}, {state_raw}" if district_raw else state_raw

    # Build state info block for frontend display
    state_info = None
    if state_data:
        state_info = {
            "state":        state_data["state"],
            "climate_zone": state_data.get("climate_zone", ""),
            "soil_type":    state_data.get("soil_type", ""),
            "major_crops":  state_data.get("major_crops", ""),
        }

    final = {
        "suitability_score": score,
        "rating":            rating,
        "rf_probability":    rf_prob,
        "ml_engine":         "Random Forest (32 crops, 3700 samples)",
        "data_source":       data_source,
        "verdict_title":     narrative.get("verdict_title", ""),
        "verdict_text":      narrative.get("verdict_text", ""),
        "detailed_advisory": narrative.get("detailed_advisory", ""),
        "rainfall":          narrative.get("rainfall_advice", ""),
        "temperature":       narrative.get("temperature_advice", ""),
        "soil":              narrative.get("soil_advice", ""),
        "best_season":       season or "Consult KVK",
        "yield_estimate":    "Refer KVK / state agriculture dept.",
        "market_potential":  "Check agmarknet.gov.in for live APMC prices",
        "factor_scores":     ml["factor_scores"],
        "alternatives":      alternatives,
        "crop_emoji":        ml_model.CROP_EMOJIS.get(crop_raw.lower(), "🌱"),
        "location_display":  location_display,
        "state":             state_raw,
        "district":          district_raw,
        "soil_inputs":       inputs,
        "state_info":        state_info,
    }

    doc = {
        "user_email":  user["email"],
        "state":       state_raw,
        "district":    district_raw,
        "location":    location_display,
        "crop":        crop_raw,
        "season":      season,
        "soil_inputs": inputs,
        "data_source": data_source,
        "result":      final,
        "created_at":  datetime.utcnow(),
    }
    ins = await db.analyses.insert_one(doc)
    return {"success": True, "analysis_id": str(ins.inserted_id), "data": final}

# ── Recommend best crop ───────────────────────────────────────────────────────
@app.post("/analysis/recommend")
async def recommend_crop(body: RecommendRequest, user=Depends(current_user)):
    top5 = ml_model.predict_crop(
        body.N, body.P, body.K, body.temperature,
        body.humidity, body.ph, body.rainfall)
    results = []
    for p in top5:
        s = ml_model.compute_suitability(
            p["crop"], body.N, body.P, body.K,
            body.temperature, body.humidity, body.ph, body.rainfall)
        results.append({
            "crop":             p["crop"].title(),
            "emoji":            p["emoji"],
            "rf_probability":   round(p["probability"] * 100, 1),
            "suitability_score":s["suitability_score"],
            "rating":           s["rating"],
        })
    loc = f"{body.district}, {body.state}" if body.district else body.state
    return {
        "success": True,
        "location": loc,
        "season":   body.season or "General",
        "soil_inputs": {
            "N": body.N, "P": body.P, "K": body.K, "ph": body.ph,
            "temperature": body.temperature, "humidity": body.humidity,
            "rainfall": body.rainfall
        },
        "recommendations": results,
    }

# ── ML info ───────────────────────────────────────────────────────────────────
@app.get("/ml/info")
async def ml_info():
    info = ml_model.get_model_info()
    return {
        "engine":          "Random Forest (sklearn)",
        "n_estimators":    300,
        "max_depth":       15,
        "dataset_source":  "Kaggle Crop Recommendation (22 crops) + synthetic extension (10 Indian crops)",
        "kaggle_url":      "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset",
        "climate_csv":     "india_state_climate.csv (32 Indian states)",
        "scoring_formula": "60pts rule-based (NPK+climate) + 40pts RF probability",
        **info,
    }

@app.get("/ml/predict")
async def ml_predict_raw(N: float, P: float, K: float, temperature: float,
                          humidity: float, ph: float, rainfall: float):
    """Quick RF test — no auth. rainfall = annual mm."""
    top5 = ml_model.predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    return {"predictions": top5, "top_crop": top5[0]["crop"] if top5 else None,
            "note": "rainfall input is annual mm, converted to seasonal internally"}

@app.get("/ml/state/{state_name}")
async def ml_state_data(state_name: str):
    """Show what climate CSV data we have for a state."""
    data = ml_model.get_state_data(state_name)
    if not data:
        raise HTTPException(404, f"State '{state_name}' not found in india_state_climate.csv")
    return {"success": True, "state_data": data,
            "note": "N/P/K are state soil health card averages; "
                    "rainfall_mm is annual IMD normal"}

# ── History ───────────────────────────────────────────────────────────────────
@app.get("/analysis/history")
async def get_history(user=Depends(current_user)):
    cur = db.analyses.find(
        {"user_email": user["email"]},
        {"_id": 1, "location": 1, "state": 1, "district": 1, "crop": 1, "season": 1,
         "result.suitability_score": 1, "result.rating": 1, "result.crop_emoji": 1,
         "result.rf_probability": 1, "created_at": 1}
    ).sort("created_at", -1).limit(20)
    records = []
    async for doc in cur:
        doc["_id"] = str(doc["_id"]); records.append(doc)
    return {"success": True, "history": records}

@app.get("/analysis/{aid}")
async def get_analysis(aid: str, user=Depends(current_user)):
    try:
        oid = ObjectId(aid)
    except:
        raise HTTPException(400, "Invalid ID")
    doc = await db.analyses.find_one({"_id": oid, "user_email": user["email"]}, {"_id": 0})
    if not doc: raise HTTPException(404, "Not found")
    return {"success": True, "data": doc}

# ── Contact ───────────────────────────────────────────────────────────────────
@app.post("/contact")
async def contact(body: ContactRequest):
    await db.contact_messages.insert_one({
        "name": body.name, "email": body.email, "subject": body.subject,
        "message": body.message, "created_at": datetime.utcnow(), "status": "unread"})
    return {"success": True, "message": "Message received."}


# ── DEBUG: see exactly what the request body contains ────────────────────────
from fastapi import Request as FastAPIRequest

@app.post("/debug/request")
async def debug_request(request: FastAPIRequest):
    """No auth. Call this to see raw request body. Remove in production."""
    body = await request.json()
    return {"received_body": body, "note": "This shows exactly what your frontend is sending"}

@app.get("/debug/state/{state_name}")
async def debug_state(state_name: str):
    """Test state lookup without auth."""
    data = ml_model.get_state_data(state_name)
    if not data:
        return {"found": False, "tried": state_name,
                "available": ml_model._climate_df["state"].tolist() if ml_model._climate_df is not None else []}
    return {"found": True, "state_data": data}

# ── Crops reference ───────────────────────────────────────────────────────────
CROP_DISPLAY = {
    "rice":      {"name":"Rice","emoji":"🌾","seasons":["Kharif"],"msps":"₹2183/quintal","yield":"4–6 t/ha"},
    "wheat":     {"name":"Wheat","emoji":"🌿","seasons":["Rabi"],"msps":"₹2275/quintal","yield":"3–5 t/ha"},
    "maize":     {"name":"Maize","emoji":"🌽","seasons":["Kharif","Rabi"],"msps":"₹2090/quintal","yield":"4–7 t/ha"},
    "cotton":    {"name":"Cotton","emoji":"🪴","seasons":["Kharif"],"msps":"₹6620/quintal","yield":"1.5–3 t/ha"},
    "sugarcane": {"name":"Sugarcane","emoji":"🎋","seasons":["Year-round"],"msps":"₹315/quintal","yield":"60–100 t/ha"},
    "soybean":   {"name":"Soybean","emoji":"🫘","seasons":["Kharif"],"msps":"₹4600/quintal","yield":"1.5–2.5 t/ha"},
    "mustard":   {"name":"Mustard","emoji":"🌻","seasons":["Rabi"],"msps":"₹5650/quintal","yield":"1–1.8 t/ha"},
    "groundnut": {"name":"Groundnut","emoji":"🥜","seasons":["Kharif"],"msps":"₹6377/quintal","yield":"1.5–3 t/ha"},
    "mango":     {"name":"Mango","emoji":"🥭","seasons":["Year-round"],"msps":"Market price","yield":"10–20 t/ha"},
    "banana":    {"name":"Banana","emoji":"🍌","seasons":["Year-round"],"msps":"Market price","yield":"30–40 t/ha"},
    "tomato":    {"name":"Tomato","emoji":"🍅","seasons":["Rabi","Zaid"],"msps":"Market price","yield":"20–30 t/ha"},
    "potato":    {"name":"Potato","emoji":"🥔","seasons":["Rabi"],"msps":"Market price","yield":"20–30 t/ha"},
    "onion":     {"name":"Onion","emoji":"🧅","seasons":["Rabi","Kharif"],"msps":"Market price","yield":"15–25 t/ha"},
    "chickpea":  {"name":"Chickpea","emoji":"🫛","seasons":["Rabi"],"msps":"₹5440/quintal","yield":"1–1.8 t/ha"},
    "lentil":    {"name":"Lentil","emoji":"🌱","seasons":["Rabi"],"msps":"₹6000/quintal","yield":"1–1.5 t/ha"},
    "jute":      {"name":"Jute","emoji":"🎍","seasons":["Kharif"],"msps":"₹5050/quintal","yield":"2–3 t/ha"},
    "coffee":    {"name":"Coffee","emoji":"☕","seasons":["Year-round"],"msps":"Market price","yield":"1–2 t/ha"},
    "coconut":   {"name":"Coconut","emoji":"🥥","seasons":["Year-round"],"msps":"Market price","yield":"60–200 nuts/tree"},
    "turmeric":  {"name":"Turmeric","emoji":"🟡","seasons":["Kharif"],"msps":"Market price","yield":"6–8 t/ha"},
    "chilli":    {"name":"Chilli","emoji":"🌶️","seasons":["Kharif","Rabi"],"msps":"Market price","yield":"1.5–3 t/ha"},
}

@app.get("/crops")
async def get_crops(search: str = ""):
    # Merge display info with ideal ranges from ml_model
    results = []
    for k, c in CROP_DISPLAY.items():
        ideal = ml_model.CROP_IDEAL.get(k, {})
        results.append({
            "key":         k,
            "name":        c["name"],
            "emoji":       c["emoji"],
            "seasons":     c["seasons"],
            "msps":        c["msps"],
            "yield_range": c["yield"],
            "ideal_rainfall":    f"{ideal.get('rainfall',(0,0))[0]}–{ideal.get('rainfall',(0,0))[1]} mm/yr" if ideal else "—",
            "ideal_temperature": f"{ideal.get('temperature',(0,0))[0]}–{ideal.get('temperature',(0,0))[1]}°C" if ideal else "—",
            "ideal_npk":         f"N:{ideal.get('N',(0,0))[0]}–{ideal.get('N',(0,0))[1]}, P:{ideal.get('P',(0,0))[0]}–{ideal.get('P',(0,0))[1]}, K:{ideal.get('K',(0,0))[0]}–{ideal.get('K',(0,0))[1]} kg/ha" if ideal else "—",
        })
    if search:
        q = search.lower()
        results = [r for r in results if q in r["name"].lower() or q in r["key"]]
    return {"success": True, "crops": results, "total": len(results)}

@app.get("/states")
async def get_states():
    if ml_model._climate_df is not None:
        states = ml_model._climate_df["state"].tolist()
    else:
        states = []
    return {"success": True, "states": states, "total": len(states),
            "source": "india_state_climate.csv"}