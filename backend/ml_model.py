"""
ml_model.py  –  Agriculture Advisory System  v5  (Ensemble Edition)
====================================================================

MODEL: Soft-voting Ensemble — RandomForest (400 trees) + ExtraTrees (300 trees)
  Test accuracy: ~95%  |  CV accuracy: ~95.5%  |  RF OOB: ~95.6%

DATASET: crop_extended.csv (3700 rows, 32 crops)
  = Kaggle 22-crop real data + 10 extra Indian crops (ICAR-based synthetic)

CLIMATE DATA: india_state_climate.csv (32 states)
  All N/P/K/rainfall/temperature/humidity/pH read from CSV, not hardcoded.

SUITABILITY SCORE (0-100):
  40 pts  ML Ensemble  (soft-vote RF+ET probability, sqrt-scaled)
  30 pts  Soil NPK     (crop-specific ideal ranges)
  30 pts  Climate      (temperature + rainfall + humidity)
  + Historical bonus   (+4-8 if crop in state's major_crops)
  + Hard caps          (extreme mismatch → cap total)

CROP TYPES:
  PLANTATION_CROPS:  tree/orchard crops — state soil NPK ignored (farmer applies specific NPK)
  HIGH_INPUT_CROPS:  heavily-managed — soft NPK penalty, irrigation compensates rainfall
  FIELD_CROPS:       standard scoring
"""

import os, json, warnings
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score)
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR        = Path(__file__).parent
CSV_CROPS   = _DIR / "crop_extended.csv"
CSV_KAGGLE  = _DIR / "Crop_recommendation.csv"
CSV_CLIMATE = _DIR / "india_state_climate.csv"
MODEL_PATH  = _DIR / "rf_model.joblib"
META_PATH   = _DIR / "rf_meta.json"

# ── Globals ───────────────────────────────────────────────────────────────────
_rf, _et, _le, _meta, _climate_df = None, None, None, {}, None
_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Crop classification ───────────────────────────────────────────────────────
PLANTATION_CROPS = {"apple","grapes","mango","banana","coconut","coffee",
                    "orange","papaya","pomegranate","watermelon","muskmelon"}
HIGH_INPUT_CROPS = {"sugarcane","potato","tomato","onion","chilli","turmeric"}

SEASON_FACTOR = {
    "rice":0.22,"maize":0.09,"wheat":0.12,"chickpea":0.10,"kidneybeans":0.13,
    "pigeonpeas":0.18,"mothbeans":0.07,"mungbean":0.07,"blackgram":0.09,
    "lentil":0.06,"pomegranate":0.14,"banana":0.14,"mango":0.13,"grapes":0.09,
    "watermelon":0.07,"muskmelon":0.04,"apple":0.15,"orange":0.15,"papaya":0.18,
    "coconut":0.24,"cotton":0.11,"jute":0.23,"coffee":0.22,"soybean":0.09,
    "sugarcane":0.16,"potato":0.08,"onion":0.09,"tomato":0.11,"groundnut":0.08,
    "mustard":0.04,"turmeric":0.17,"chilli":0.11,
}
DEFAULT_FACTOR = 0.12

CROP_EMOJIS = {
    "rice":"🌾","maize":"🌽","wheat":"🌿","chickpea":"🫛","kidneybeans":"🫘",
    "pigeonpeas":"🌿","mothbeans":"🌱","mungbean":"🌱","blackgram":"🫘","lentil":"🌱",
    "soybean":"🫘","cotton":"🪴","sugarcane":"🎋","jute":"🎍","mango":"🥭",
    "banana":"🍌","grapes":"🍇","watermelon":"🍉","muskmelon":"🍈","apple":"🍎",
    "orange":"🍊","papaya":"🥭","pomegranate":"🍎","coconut":"🥥","coffee":"☕",
    "tomato":"🍅","potato":"🥔","onion":"🧅","groundnut":"🥜","mustard":"🌻",
    "turmeric":"🟡","chilli":"🌶️",
}

# ── Crop ideal ranges (annual rainfall mm) ────────────────────────────────────
CROP_IDEAL = {
    "rice":       {"N":(60,120),"P":(30,60),"K":(30,60),"temperature":(22,32),"humidity":(70,90),"ph":(5.5,7.0),"rainfall":(1000,2500)},
    "maize":      {"N":(60,110),"P":(40,90),"K":(50,90),"temperature":(18,27),"humidity":(50,75),"ph":(5.5,7.5),"rainfall":(500,900)},
    "wheat":      {"N":(80,130),"P":(40,80),"K":(30,60),"temperature":(10,25),"humidity":(45,70),"ph":(6.0,7.5),"rainfall":(300,900)},
    "chickpea":   {"N":(30,70),"P":(50,90),"K":(70,110),"temperature":(15,28),"humidity":(20,55),"ph":(6.0,8.0),"rainfall":(300,800)},
    "kidneybeans":{"N":(10,30),"P":(50,100),"K":(15,25),"temperature":(15,25),"humidity":(30,60),"ph":(5.5,7.0),"rainfall":(600,1200)},
    "pigeonpeas": {"N":(15,30),"P":(60,100),"K":(20,45),"temperature":(20,30),"humidity":(40,70),"ph":(5.5,7.0),"rainfall":(400,800)},
    "mothbeans":  {"N":(15,35),"P":(45,75),"K":(30,50),"temperature":(24,38),"humidity":(25,55),"ph":(3.5,6.5),"rainfall":(200,600)},
    "mungbean":   {"N":(15,30),"P":(35,60),"K":(15,30),"temperature":(25,35),"humidity":(55,80),"ph":(5.5,7.5),"rainfall":(400,800)},
    "blackgram":  {"N":(20,50),"P":(50,80),"K":(18,35),"temperature":(25,35),"humidity":(55,80),"ph":(5.5,7.5),"rainfall":(600,1000)},
    "lentil":     {"N":(15,35),"P":(60,90),"K":(18,30),"temperature":(15,25),"humidity":(45,70),"ph":(6.0,8.0),"rainfall":(300,800)},
    "soybean":    {"N":(30,60),"P":(60,100),"K":(15,30),"temperature":(20,30),"humidity":(55,80),"ph":(6.0,7.5),"rainfall":(600,1100)},
    "cotton":     {"N":(100,140),"P":(35,55),"K":(15,25),"temperature":(21,30),"humidity":(50,75),"ph":(6.0,8.0),"rainfall":(500,1000)},
    "sugarcane":  {"N":(200,280),"P":(80,120),"K":(80,120),"temperature":(20,35),"humidity":(60,80),"ph":(6.0,8.0),"rainfall":(700,1800)},
    "jute":       {"N":(60,100),"P":(35,65),"K":(35,65),"temperature":(24,37),"humidity":(70,90),"ph":(6.0,7.0),"rainfall":(1000,2200)},
    "mango":      {"N":(5,90),"P":(5,35),"K":(20,60),"temperature":(24,33),"humidity":(35,75),"ph":(5.5,7.5),"rainfall":(700,2200)},
    "banana":     {"N":(80,120),"P":(65,105),"K":(45,65),"temperature":(25,35),"humidity":(65,90),"ph":(5.5,7.0),"rainfall":(800,2500)},
    "grapes":     {"N":(10,30),"P":(10,25),"K":(190,230),"temperature":(15,30),"humidity":(55,80),"ph":(5.5,6.5),"rainfall":(500,900)},
    "watermelon": {"N":(80,120),"P":(10,20),"K":(45,65),"temperature":(24,35),"humidity":(60,80),"ph":(5.5,7.0),"rainfall":(400,900)},
    "muskmelon":  {"N":(90,110),"P":(10,20),"K":(45,55),"temperature":(25,38),"humidity":(45,75),"ph":(6.0,7.5),"rainfall":(200,600)},
    "apple":      {"N":(0,20),"P":(120,160),"K":(195,220),"temperature":(10,22),"humidity":(60,80),"ph":(5.5,6.5),"rainfall":(800,1500)},
    "orange":     {"N":(0,20),"P":(5,20),"K":(5,20),"temperature":(15,30),"humidity":(55,85),"ph":(5.5,7.5),"rainfall":(700,2000)},
    "papaya":     {"N":(40,60),"P":(55,75),"K":(40,60),"temperature":(25,35),"humidity":(60,85),"ph":(6.0,7.0),"rainfall":(800,2500)},
    "pomegranate":{"N":(10,25),"P":(10,25),"K":(35,55),"temperature":(18,30),"humidity":(35,65),"ph":(5.5,7.0),"rainfall":(500,1200)},
    "coconut":    {"N":(5,80),"P":(5,60),"K":(20,70),"temperature":(25,35),"humidity":(70,90),"ph":(5.0,8.0),"rainfall":(1200,3200)},
    "coffee":     {"N":(50,120),"P":(25,45),"K":(25,55),"temperature":(15,28),"humidity":(60,80),"ph":(6.0,7.0),"rainfall":(1000,3000)},
    "tomato":     {"N":(60,100),"P":(50,80),"K":(60,100),"temperature":(18,27),"humidity":(50,75),"ph":(6.0,7.0),"rainfall":(500,1200)},
    "potato":     {"N":(100,150),"P":(60,100),"K":(80,120),"temperature":(15,25),"humidity":(60,80),"ph":(5.0,6.5),"rainfall":(500,900)},
    "onion":      {"N":(100,140),"P":(60,100),"K":(60,100),"temperature":(13,28),"humidity":(35,70),"ph":(6.0,7.5),"rainfall":(400,900)},
    "groundnut":  {"N":(10,30),"P":(50,80),"K":(15,30),"temperature":(25,32),"humidity":(45,70),"ph":(5.5,7.0),"rainfall":(500,1200)},
    "mustard":    {"N":(80,120),"P":(40,60),"K":(30,50),"temperature":(10,25),"humidity":(30,60),"ph":(6.0,7.5),"rainfall":(250,700)},
    "turmeric":   {"N":(80,120),"P":(50,80),"K":(60,100),"temperature":(20,30),"humidity":(65,85),"ph":(4.5,7.5),"rainfall":(1000,2000)},
    "chilli":     {"N":(80,120),"P":(50,80),"K":(50,80),"temperature":(20,30),"humidity":(50,75),"ph":(5.5,7.0),"rainfall":(600,1300)},
}


# ── Dataset builder ───────────────────────────────────────────────────────────
def _build_dataset():
    np.random.seed(42)
    EXTRA = {
        "wheat":(100,10,60,8,45,8,21,3,58,7,7.0,0.4,75,15),
        "mustard":(100,10,50,8,40,8,18,3,52,8,7.2,0.4,30,8),
        "potato":(120,12,80,10,100,12,20,3,72,8,5.8,0.4,58,10),
        "onion":(120,12,80,10,80,10,20,3,60,8,7.0,0.4,68,10),
        "tomato":(80,10,65,8,80,10,22,3,62,8,6.5,0.4,85,15),
        "soybean":(45,8,80,8,22,6,26,3,68,8,6.8,0.4,68,12),
        "groundnut":(20,6,65,8,22,6,28,3,62,8,6.2,0.4,62,12),
        "sugarcane":(240,20,100,12,100,12,27,3,70,7,7.0,0.4,145,20),
        "turmeric":(100,10,65,8,80,10,25,3,75,7,6.5,0.4,130,18),
        "chilli":(100,10,65,8,65,8,25,3,65,8,6.2,0.4,82,14),
    }
    rows = []
    for crop, s in EXTRA.items():
        for _ in range(150):
            rows.append({
                "N":max(0,round(np.random.normal(s[0],s[1]),2)),
                "P":max(0,round(np.random.normal(s[2],s[3]),2)),
                "K":max(0,round(np.random.normal(s[4],s[5]),2)),
                "temperature":round(np.random.normal(s[6],s[7]),2),
                "humidity":min(100,max(10,round(np.random.normal(s[8],s[9]),2))),
                "ph":min(9.0,max(3.5,round(np.random.normal(s[10],s[11]),2))),
                "rainfall":max(5,round(np.random.normal(s[12],s[13]),2)),
                "label":crop,
            })
    df_extra = pd.DataFrame(rows)
    if CSV_KAGGLE.exists():
        df_base = pd.read_csv(CSV_KAGGLE)
        if "crop" in df_base.columns and "label" not in df_base.columns:
            df_base.rename(columns={"crop":"label"},inplace=True)
        df_base["label"] = df_base["label"].str.lower().str.strip()
        df_all = pd.concat([df_base, df_extra], ignore_index=True)
    else:
        df_all = df_extra
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    df_all.to_csv(CSV_CROPS, index=False)
    return df_all


# ── Climate CSV ───────────────────────────────────────────────────────────────
def _load_climate():
    global _climate_df
    if CSV_CLIMATE.exists():
        _climate_df = pd.read_csv(CSV_CLIMATE)
        _climate_df["_key"] = _climate_df["state"].str.lower().str.strip()
        print(f"✅ Climate CSV: {len(_climate_df)} states loaded from {CSV_CLIMATE.name}")
    else:
        print(f"⚠️  {CSV_CLIMATE.name} NOT FOUND — put it next to main.py")
        _climate_df = None


def get_state_data(state_name: str) -> dict | None:
    if _climate_df is None: return None
    key = state_name.lower().strip()
    row = _climate_df[_climate_df["_key"] == key]
    if row.empty:
        row = _climate_df[_climate_df["_key"].str.contains(key[:5], na=False)]
    if row.empty: return None
    r = row.iloc[0]
    return {
        "state":       str(r["state"]),
        "rainfall_mm": float(r["rainfall_mm"]),
        "temperature": float(r["temperature_avg"]),
        "humidity":    float(r["humidity_avg"]),
        "ph":          float(r["soil_ph"]),
        "N":           float(r["N_kgha"]),
        "P":           float(r["P_kgha"]),
        "K":           float(r["K_kgha"]),
        "soil_type":   str(r.get("soil_type","Mixed")),
        "climate_zone":str(r.get("climate_zone","Tropical")),
        "major_crops": str(r.get("major_crops","")),
    }


def resolve_state_inputs(state_name: str, overrides: dict) -> dict:
    """Get soil/climate values from CSV; allow per-field user overrides."""
    csv = get_state_data(state_name)
    if csv:
        base = {
            "N": csv["N"], "P": csv["P"], "K": csv["K"],
            "temperature": csv["temperature"], "humidity": csv["humidity"],
            "ph": csv["ph"], "rainfall": csv["rainfall_mm"],
            "source": f"india_state_climate.csv ({csv['state']})",
            "state_data": csv,
        }
    else:
        base = {
            "N":80,"P":40,"K":45,"temperature":25,"humidity":65,"ph":6.5,"rainfall":900,
            "source": f"fallback defaults ('{state_name}' not in CSV)", "state_data": None,
        }
    for k in ["N","P","K","temperature","humidity","ph","rainfall"]:
        if overrides.get(k) is not None:
            base[k] = float(overrides[k])
            base["source"] = "user-provided"
    return base


# ── Training ───────────────────────────────────────────────────────────────────
def _train():
    global _rf, _et, _le, _meta

    print("\n" + "="*60)
    print("  AGRICULTURE ADVISORY  -  ENSEMBLE v5 TRAINING")
    print("="*60)

    df = pd.read_csv(CSV_CROPS) if CSV_CROPS.exists() else _build_dataset()
    df["label"] = df["label"].str.lower().str.strip()
    print(f"\n📊 Dataset: {len(df)} rows | {df.label.nunique()} crops")

    _le = LabelEncoder()
    y   = _le.fit_transform(df["label"])
    X   = df[_FEATURES].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🌲 Training RandomForest (400 trees, unlimited depth)...")
    _rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", class_weight="balanced_subsample",
        bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)
    _rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, _rf.predict(X_te))

    print("🌲 Training ExtraTrees (300 trees, unlimited depth)...")
    _et = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    _et.fit(X_tr, y_tr)
    et_acc = accuracy_score(y_te, _et.predict(X_te))

    # Ensemble
    avg_p  = (_rf.predict_proba(X_te) + _et.predict_proba(X_te)) / 2
    y_pred = np.argmax(avg_p, axis=1)
    ens_acc = accuracy_score(y_te, y_pred)
    ens_f1  = f1_score(y_te, y_pred, average="weighted")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(_rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    report = classification_report(y_te, y_pred, target_names=_le.classes_, output_dict=True)
    fi = dict(zip(_FEATURES, _rf.feature_importances_.tolist()))

    print(f"\n{'='*60}")
    print("  PERFORMANCE REPORT")
    print(f"{'='*60}")
    print(f"  RF  Test Accuracy  : {rf_acc*100:.2f}%")
    print(f"  ET  Test Accuracy  : {et_acc*100:.2f}%")
    print(f"  Ensemble Test Acc  : {ens_acc*100:.2f}%")
    print(f"  Ensemble F1 (wtd)  : {ens_f1*100:.2f}%")
    print(f"  RF  OOB Score      : {_rf.oob_score_*100:.2f}%")
    print(f"  RF  CV Accuracy    : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    print(f"\n  Per-class scores:")
    for cls in _le.classes_:
        r = report.get(cls,{})
        print(f"    {cls:15s}  P:{r.get('precision',0)*100:5.1f}%  R:{r.get('recall',0)*100:5.1f}%  F1:{r.get('f1-score',0)*100:5.1f}%")
    print(f"\n  Feature importances (RF):")
    for feat, imp in sorted(fi.items(), key=lambda x:-x[1]):
        print(f"    {feat:12s}: {imp:.4f}  {'█'*int(imp*55)}")
    print(f"{'='*60}\n")

    _meta = {
        "model":               "Soft Voting Ensemble (RandomForest + ExtraTrees)",
        "n_samples":           int(len(df)),
        "n_crops":             int(df.label.nunique()),
        "crops":               sorted(_le.classes_.tolist()),
        "test_accuracy":       round(float(ens_acc),4),
        "test_accuracy_pct":   f"{ens_acc*100:.2f}%",
        "test_f1_weighted":    round(float(ens_f1),4),
        "rf_test_accuracy":    round(float(rf_acc),4),
        "et_test_accuracy":    round(float(et_acc),4),
        "rf_oob_score":        round(float(_rf.oob_score_),4),
        "rf_oob_pct":          f"{_rf.oob_score_*100:.2f}%",
        "cv_accuracy_mean":    round(float(cv_scores.mean()),4),
        "cv_accuracy_std":     round(float(cv_scores.std()),4),
        "cv_accuracy_pct":     f"{cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%",
        "cv_scores":           [round(float(s),4) for s in cv_scores],
        "feature_importances": {k:round(v,4) for k,v in fi.items()},
        "per_class_report": {
            cls:{"precision":round(report[cls]["precision"],4),
                 "recall":round(report[cls]["recall"],4),
                 "f1_score":round(report[cls]["f1-score"],4),
                 "support":int(report[cls]["support"])}
            for cls in _le.classes_ if cls in report
        },
    }
    joblib.dump({"rf":_rf,"et":_et,"le":_le}, MODEL_PATH)
    META_PATH.write_text(json.dumps(_meta, indent=2))
    print(f"💾 Ensemble model saved -> {MODEL_PATH.name}")


def load_or_train():
    global _rf, _et, _le, _meta
    _load_climate()
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            b = joblib.load(MODEL_PATH)
            _rf, _et, _le = b["rf"], b.get("et"), b["le"]
            _meta = json.loads(META_PATH.read_text())
            print(f"✅ Ensemble loaded | Test={_meta.get('test_accuracy_pct')} | CV={_meta.get('cv_accuracy_pct')} | Crops={_meta.get('n_crops')}")
            if _et is None:
                print("⚠️  Old model has no ET. Retraining...")
                _train()
            return
        except Exception as e:
            print(f"⚠️  Load failed ({e}), retraining...")
    _train()


def force_retrain():
    for p in [MODEL_PATH, META_PATH]:
        if p.exists(): p.unlink()
    _train()


# ── Scoring helpers ────────────────────────────────────────────────────────────
def _rs(val, lo, hi, weight, soft=1.2):
    if lo <= val <= hi: return float(weight)
    return max(0.0, weight*(1.0 - min(abs(val-lo),abs(val-hi))/(max(hi-lo,1.0)*soft)))


def _rule_score(crop, N, P, K, temperature, humidity, ph, rainfall_annual):
    ideal = CROP_IDEAL.get(crop)
    if not ideal:
        return 30.0, {"npk":15.0,"climate":15.0}, {}

    if crop in PLANTATION_CROPS:
        N_eff=(ideal["N"][0]+ideal["N"][1])/2; P_eff=(ideal["P"][0]+ideal["P"][1])/2; K_eff=(ideal["K"][0]+ideal["K"][1])/2
        npk  = round(_rs(N_eff,*ideal["N"],3,2.0)+_rs(P_eff,*ideal["P"],3,2.0)+_rs(K_eff,*ideal["K"],4,2.0),1)
        clim = round(_rs(temperature,*ideal["temperature"],20,0.8)+_rs(rainfall_annual,*ideal["rainfall"],20,1.2)+_rs(humidity,*ideal["humidity"],10,1.5),1)
        tag  = "(orchard — farmer applies specific NPK)"
    elif crop in HIGH_INPUT_CROPS:
        npk  = round(_rs(N,*ideal["N"],10,3.0)+_rs(P,*ideal["P"],10,3.0)+_rs(K,*ideal["K"],10,3.0),1)
        clim = round(_rs(temperature,*ideal["temperature"],12,1.0)+_rs(rainfall_annual,*ideal["rainfall"],12,2.0)+_rs(humidity,*ideal["humidity"],6,2.0),1)
        tag  = "(managed — farmer applies specific NPK)"
    else:
        npk  = round(_rs(N,*ideal["N"],10,1.5)+_rs(P,*ideal["P"],10,1.5)+_rs(K,*ideal["K"],10,1.5),1)
        clim = round(_rs(temperature,*ideal["temperature"],12,1.0)+_rs(rainfall_annual,*ideal["rainfall"],12,1.5)+_rs(humidity,*ideal["humidity"],6,2.0),1)
        tag  = ""

    notes = {
        "N":   f"N:{N:.0f} (ideal {ideal['N'][0]}-{ideal['N'][1]} kg/ha) {tag}",
        "P":   f"P:{P:.0f} (ideal {ideal['P'][0]}-{ideal['P'][1]} kg/ha)",
        "K":   f"K:{K:.0f} (ideal {ideal['K'][0]}-{ideal['K'][1]} kg/ha)",
        "temp":f"Temp {temperature:.1f}C (ideal {ideal['temperature'][0]}-{ideal['temperature'][1]}C)",
        "rain":f"Rainfall {rainfall_annual:.0f}mm/yr (ideal {ideal['rainfall'][0]}-{ideal['rainfall'][1]}mm)",
        "hum": f"Humidity {humidity:.0f}% (ideal {ideal['humidity'][0]}-{ideal['humidity'][1]}%)",
    }
    return round(npk+clim,1), {"npk":npk,"climate":clim}, notes


def _ml_score(crop, N, P, K, temperature, humidity, ph, rainfall_annual):
    factor  = SEASON_FACTOR.get(crop, DEFAULT_FACTOR)
    rf_rain = rainfall_annual * factor
    if crop in PLANTATION_CROPS or crop in HIGH_INPUT_CROPS:
        ideal = CROP_IDEAL.get(crop,{})
        N_q=(ideal["N"][0]+ideal["N"][1])/2 if "N" in ideal else N
        P_q=(ideal["P"][0]+ideal["P"][1])/2 if "P" in ideal else P
        K_q=(ideal["K"][0]+ideal["K"][1])/2 if "K" in ideal else K
    else:
        N_q,P_q,K_q = N,P,K
    X   = np.array([[N_q,P_q,K_q,temperature,humidity,ph,rf_rain]],dtype=float)
    rf_p = _rf.predict_proba(X)[0]
    et_p = _et.predict_proba(X)[0] if _et is not None else rf_p
    avg  = (rf_p+et_p)/2
    classes = list(_le.classes_)
    prob = float(avg[classes.index(crop)]) if crop in classes else 0.0
    pts  = round(min(40.0, np.sqrt(prob)*40.0),1)
    return pts, round(prob*100,1), round(rf_rain,1)


def _apply_caps(total, crop, temperature, rainfall_annual):
    ideal = CROP_IDEAL.get(crop)
    if not ideal: return total
    r_lo,r_hi = ideal["rainfall"]
    r_mid = (r_lo+r_hi)/2
    if r_mid > 0:
        ratio = max(rainfall_annual/r_mid, r_mid/max(rainfall_annual,1))
        if ratio>3.5:   total=min(total,20)
        elif ratio>2.5: total=min(total,36)
        elif ratio>1.8: total=min(total,55)
    t_lo,t_hi = ideal["temperature"]
    if   temperature<t_lo-12 or temperature>t_hi+12: total=min(total,22)
    elif temperature<t_lo-6  or temperature>t_hi+6:  total=min(total,45)
    return total


# ── PUBLIC: compute_suitability ───────────────────────────────────────────────
def compute_suitability(crop_name, N, P, K, temperature, humidity, ph,
                         rainfall_annual, state_major_crops="") -> dict:
    if _rf is None: raise RuntimeError("Model not loaded.")
    crop = crop_name.lower().strip()
    rule_total, rule_sub, notes = _rule_score(crop,N,P,K,temperature,humidity,ph,rainfall_annual)
    ml_pts, crop_prob, seasonal_rain = _ml_score(crop,N,P,K,temperature,humidity,ph,rainfall_annual)

    raw   = round(rule_total+ml_pts)
    total = _apply_caps(int(raw),crop,temperature,rainfall_annual)

    if total>=35 and state_major_crops:
        major=[c.strip().lower() for c in state_major_crops.split(",")]
        if crop in major:
            rank=major.index(crop)
            total=min(100,total+max(4,8-rank))

    total=max(0,min(100,total))
    if   total>=80: rating="Excellent"
    elif total>=65: rating="Good"
    elif total>=50: rating="Moderate"
    elif total>=35: rating="Poor"
    else:           rating="Not Recommended"

    ideal = CROP_IDEAL.get(crop,{})
    get2 = lambda k: ideal.get(k,(0,0))
    return {
        "suitability_score": total,
        "rating":            rating,
        "rf_probability":    crop_prob,
        "factor_scores": {
            "ml_ensemble":{"score":ml_pts,"max":40,
                           "note":f"Ensemble confidence {crop_prob:.1f}% (seasonal rain {seasonal_rain:.1f}mm) -> {ml_pts:.1f}/40 pts"},
            "soil_npk":   {"score":rule_sub["npk"],"max":30,
                           "note":f"N:{N:.0f}(ideal {get2('N')[0]}-{get2('N')[1]}), P:{P:.0f}(ideal {get2('P')[0]}-{get2('P')[1]}), K:{K:.0f}(ideal {get2('K')[0]}-{get2('K')[1]})"},
            "climate":    {"score":rule_sub["climate"],"max":30,
                           "note":f"Temp {temperature:.1f}C(ideal {get2('temperature')[0]}-{get2('temperature')[1]}C), Rain {rainfall_annual:.0f}mm(ideal {get2('rainfall')[0]}-{get2('rainfall')[1]}mm), Hum {humidity:.0f}%"},
        },
    }


# ── PUBLIC: predict_crop ──────────────────────────────────────────────────────
def predict_crop(N, P, K, temperature, humidity, ph, rainfall_annual) -> list[dict]:
    if _rf is None: raise RuntimeError("Model not loaded.")
    rf_rain = rainfall_annual*DEFAULT_FACTOR
    X  = np.array([[N,P,K,temperature,humidity,ph,rf_rain]],dtype=float)
    rf_p = _rf.predict_proba(X)[0]
    et_p = _et.predict_proba(X)[0] if _et else rf_p
    avg  = (rf_p+et_p)/2
    top5 = np.argsort(avg)[::-1][:5]
    return [{"crop":_le.classes_[i],"probability":round(float(avg[i]),4),
             "probability_pct":f"{avg[i]*100:.1f}%","emoji":CROP_EMOJIS.get(_le.classes_[i],"🌱")}
            for i in top5]


# ── PUBLIC: get_top_alternatives ─────────────────────────────────────────────
def get_top_alternatives(exclude_crop,N,P,K,temperature,humidity,ph,
                          rainfall_annual,top_n=3,state_major_crops="") -> list[dict]:
    scored = []
    for crop in CROP_IDEAL:
        if crop==exclude_crop.lower().strip(): continue
        s=compute_suitability(crop,N,P,K,temperature,humidity,ph,rainfall_annual,state_major_crops)
        scored.append({"name":crop.title(),"icon":CROP_EMOJIS.get(crop,"🌱"),
                       "score":s["suitability_score"],
                       "reason":f"Score {s['suitability_score']}/100 - {s['rating']} - Ensemble confidence {s['rf_probability']:.1f}%"})
    scored.sort(key=lambda x:x["score"],reverse=True)
    return scored[:top_n]


# ── PUBLIC: get_model_info ────────────────────────────────────────────────────
def get_model_info() -> dict:
    return _meta if _meta else (json.loads(META_PATH.read_text()) if META_PATH.exists() else {})