from fastapi import FastAPI, File, UploadFile, Form  # type: ignore
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # type: ignore
import joblib  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np
from typing import Optional
import pandas as pd
import re
import tempfile
import shutil
import subprocess
import json
import unicodedata as ud
from sklearn.metrics.pairwise import cosine_similarity

#initialize app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  #to match the frontend 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#to standardize apps
from pathlib import Path
import os as _os
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

def _dir_from_env(name, default):
    return Path(_os.getenv(name, str(default))).resolve()

MODELS_DIR = _dir_from_env("MODELS_DIR", PROJECT_ROOT / "models")
ANALYSIS_DIR = _dir_from_env("ANALYSIS_DIR", PROJECT_ROOT / "analysis")
DATA_DIR = _dir_from_env("DATA_DIR", PROJECT_ROOT / "data")


#models
LONG_MODEL_PATH = str((MODELS_DIR / "long_model_under1_attempt.pkl").resolve())
SHORT_MODEL_PATH = str((MODELS_DIR / "short_videos_model.pkl").resolve())
long_model = joblib.load(LONG_MODEL_PATH)
short_model = joblib.load(SHORT_MODEL_PATH)

#load the sentence transformer like traniing
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#regex
_url_pat = re.compile(r"https?://\S+|www\.\S+")
_ctrl_pat = re.compile(r"[\r\n\t\0\u200B-\u200D\uFEFF]")
_ws_pat = re.compile(r"\s+")

#helper functions blabla
def normalise_text(txt: str | float) -> str:
    if pd.isna(txt):
        return ""
    txt = str(txt)
    txt = _url_pat.sub("", txt)
    txt = _ctrl_pat.sub(" ", txt)
    txt = ud.normalize("NFKC", txt)
    txt = _ws_pat.sub(" ", txt).strip()
    return txt

class VideoData(BaseModel):
    video_type: str
    title: str
    description: str
    keywords: Optional[str] = ""
    duration: float

def get_video_duration(file_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

#to actally create the features like in training for long
def create_features_long(data: VideoData):
    title_norm = normalise_text(data.title)
    desc_norm = normalise_text(data.description)
    keywords_norm = normalise_text(data.keywords)

    df = pd.DataFrame([{
        "title": title_norm,
        "description": desc_norm,
        "keywords": keywords_norm,
        "duration": data.duration,
    }])

    df["title_len"] = df["title"].str.len()
    df["has_qmark"] = df["title"].str.contains(r"\?").astype("int8")
    df["has_excl"] = df["title"].str.contains(r"!").astype("int8")
    df["has_top_num"] = df["title"].str.contains(r"\btop\s+\d+", flags=re.I).astype("int8")
    df["caps_ratio"] = (df["title"].str.count(r"[A-Z]") / df["title_len"].clip(lower=1)).clip(0, 1)
    df["punct_ratio"] = (df["title"].str.count(r"[.!?]") / df["title_len"].clip(lower=1)).clip(0, 1)
    df["title_word_count"] = df["title"].str.split().str.len()
    df["is_short_title"] = (df["title_word_count"] <= 3).astype("int8")

    clickbait_words = ["you", "your", "top", "best", "insane", "crazy", "shocking", "don’t", "secret"]
    df["clickbait_score"] = df["title"].apply(lambda t: sum(1 for w in clickbait_words if w in t.lower()))

    df["desc_len"] = df["description"].str.len()
    df["desc_word_count"] = df["description"].str.split().str.len()
    url_pat = re.compile(r"https?://")
    df["url_count"] = df["description"].str.count(url_pat)

    def count_keyword_overlap(row):
        desc = normalise_text(row["description"]).lower()
        kws = normalise_text(row["keywords"]).lower()
        kws_clean = re.sub(r"[,\s]+", " ", kws).strip()
        desc_clean = re.sub(r"[,\s]+", " ", desc).strip()
        return len(set(desc_clean.split()).intersection(set(kws_clean.split())))


    df["desc_kw_overlap"] = df.apply(count_keyword_overlap, axis=1)
    df["duration_min"] = df["duration"] / 60.0
    df["duration_log"] = np.log1p(df["duration"])
    df["kw_count"] = len(keywords_norm.split()) if keywords_norm.strip() else 0
    keywords_cleaned = re.sub(r"[,\s]+", " ", keywords_norm).strip()
    kw_tokens = keywords_cleaned.split()
    df["kw_count_log"] = np.log1p(len(kw_tokens)) if kw_tokens else 0.0

    bins = [0, 300, 600, 1200, 1800, 3600]
    labels = ["XS", "S", "M", "L", "XL"]
    df["duration_cat"] = pd.cut(df["duration"], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=["duration_cat"])
    for col in ["duration_cat_XS", "duration_cat_S", "duration_cat_M", "duration_cat_L", "duration_cat_XL"]:
        if col not in df.columns:
            df[col] = 0

    placeholder_pat = re.compile(r"\[(?:MISSING|NO)[^\]]*\]", flags=re.I)
    for col in ["title", "description"]:
        has_ph = df[col].str.contains(placeholder_pat, regex=True, na=False)
        df.loc[has_ph, col] = ""
        flag_col = f"{col.split('_')[0]}_missing"
        df[flag_col] = (df[col] == "").astype("int8")

    cols_to_drop = [
        'has_qmark', 'has_top_num', 'url_count', 'title_missing',
        'kw_count', 'clickbait_score', 'is_short_title', 'title_word_count',
        'title_len', 'duration_cat_S', 'duration_cat_XS',
        'description_missing', 'caps_ratio'
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in ["title", "description", "keywords"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_features = df.to_numpy()

    # FIX: reshape embeddings
    desc_emb = embedder.encode([desc_norm])[0].reshape(1, -1)
    title_emb = embedder.encode([title_norm])[0].reshape(1, -1)
    kw_clean = re.sub(r"[,\s]+", " ", keywords_norm).strip()
    keywords_emb = embedder.encode([kw_clean])[0].reshape(1, -1)

    final_features = np.hstack([numeric_features, desc_emb, title_emb, keywords_emb])
    return final_features.reshape(1, -1)


#same here but for short
def create_features_short(data: VideoData):
    title_norm = normalise_text(data.title)
    desc_norm = normalise_text(data.description)

    df = pd.DataFrame([{
        "title": title_norm,
        "description": desc_norm,
        "duration": data.duration,
    }])

    df["title_len"] = df["title"].str.len()
    df["desc_len"] = df["description"].str.len()
    df["title_word_count"] = df["title"].apply(lambda x: len(str(x).split()))
    df["desc_word_count"] = df["description"].apply(lambda x: len(str(x).split()))
    df["title_unique_words"] = df["title"].apply(lambda x: len(set(str(x).split())))
    df["desc_unique_words"] = df["description"].apply(lambda x: len(set(str(x).split())))

    stopwords = {"the","and","a","to","of","in","is","you","that","it",
                 "on","for","with","as","at","by","an"}
    df["title_stopword_ratio"] = df["title"].apply(
        lambda x: sum(w.lower() in stopwords for w in str(x).split()) / max(1, len(str(x).split()))
    )
    df["desc_stopword_ratio"] = df["description"].apply(
        lambda x: sum(w.lower() in stopwords for w in str(x).split()) / max(1, len(str(x).split()))
    )

    df["has_qmark"] = df["title"].str.contains(r"\?").astype("int8")
    df["has_excl"] = df["title"].str.contains(r"!").astype("int8")
    df["caps_ratio"] = (df["title"].str.count(r"[A-Z]") / df["title_len"].clip(lower=1)).clip(0, 1)
    df["punct_ratio"] = (df["title"].str.count(r"[.!?]") / df["title_len"].clip(lower=1)).clip(0, 1)

    df["duration_log"] = np.log1p(df["duration"])
    bins = [0, 15, 30, 45, 60, np.inf]
    labels = ["XS", "S", "M", "L", "XL"]
    df["duration_cat"] = pd.cut(df["duration"], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=["duration_cat"])
    df["duration_norm"] = df["duration"] / 60.0
    for col in ["duration_cat_XS", "duration_cat_S", "duration_cat_M", "duration_cat_L", "duration_cat_XL"]:
        if col not in df.columns:
            df[col] = 0

    df["title_desc_len_ratio"] = (
        df["title_len"] / df["desc_len"].replace(0, np.nan)
    ).fillna(0).clip(0, 5)

    df["title_missing"] = df["title"].str.strip().eq("").astype(int)
    df["title_synthetic"] = df["title_missing"].astype(int)
    df["description_missing"] = df["description"].str.strip().eq("").astype(int)

    drop_cols = ["title", "description", "views", "views_log", "video_id", "title_missing", "title_synthetic"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_features = df.to_numpy()

    # FIX: reshape embeddings
    title_emb = embedder.encode([title_norm])[0].reshape(1, -1)
    desc_emb = embedder.encode([desc_norm])[0].reshape(1, -1)

    final_features = np.hstack([numeric_features, title_emb, desc_emb])
    return final_features.reshape(1, -1)

#stuff to help transalte for feedabck later
FEATURE_LABELS = {
    "duration_min": "Durée de la vidéo (minutes)",
    "desc_len": "Longueur de la description (en caractères)",
    "title_desc_len_ratio": "Rapport longueur titre/description",
    "title_stopword_ratio": "Ratio de mots vides dans le titre"
}

TEXT_FIELD_LABELS = {
    "Title": "Titre",
    "Description": "Description",
    "Keywords": "Mots-clés"
}

#embedder
def _encode(text: str) -> np.ndarray:
    return embedder.encode([text])[0]

#important features here
with open((PROJECT_ROOT / "frontend" / "feature_impact_long.json"), "r") as f:
    feature_impact_long = json.load(f)
with open((PROJECT_ROOT / "frontend" / "feature_impact_short.json"), "r") as f:
    feature_impact_short = json.load(f)

#avg embeddings
avg_long_title = np.load(str((ANALYSIS_DIR / 'long_title_avg.npy').resolve()))
avg_long_desc = np.load(str((ANALYSIS_DIR / 'long_description_avg.npy').resolve()))
avg_long_keywords = np.load(str((ANALYSIS_DIR / 'long_keywords_avg.npy').resolve()))

avg_short_title = np.load(str((ANALYSIS_DIR / 'short_title_avg.npy').resolve()))
avg_short_desc = np.load(str((ANALYSIS_DIR / 'short_description_avg.npy').resolve()))

#feeedback for similarity wvcwbdc im so done with this
def _diff_message(label: str, diff: float, is_ratio: bool = False) -> str:
    if is_ratio:
        status = "plus élevé" if diff > 0 else "plus bas"
        return f"{label} est {status} que la moyenne des vidéos performantes de {abs(diff):.3f}"
    else:
        status = "plus long" if diff > 0 else "plus court"
        return f"{label} est {status} que la moyenne des vidéos performantes de {abs(diff):.2f}"

def _sim_message(similarity: float) -> str:
    if similarity >= 0.85:
        return "Très similaire aux vidéos les plus performantes"
    elif similarity >= 0.7:
        return "Assez similaire aux vidéos performantes"
    elif similarity >= 0.4:
        return "Différent, mais potentiellement intéressant"
    else:
        return "Assez différent des vidéos les plus performantes"

#some actual feedback:
def _duration_message(label: str, diff: float) -> str:
    minutes = abs(round(diff))
    if abs(diff) < 1:
        return f"{label} est très proche de la moyenne des vidéos les plus performantes."
    elif diff > 1:
        return f"Votre vidéo est plus longue que la moyenne des vidéos performantes. Essayez de retirer environ {minutes} minute(s)."
    else:
        return f"Votre vidéo est plus courte que la moyenne des vidéos performantes. Essayez d’ajouter environ {minutes} minute(s)."

def _generic_message(label: str, diff: float, is_ratio: bool = False) -> str:
    if abs(diff) < 10:
        return f"{label} est similaire à la moyenne des vidéos performantes."
    elif diff > 0:
        return f"{label} est plus élevée que la moyenne des vidéos performantes de {round(diff)}."
    else:
        return f"{label} est plus basse que la moyenne des vidéos performantes de {round(abs(diff))}."

def _ratio_message(label: str, diff: float) -> str:
    if abs(diff) < 0.1:
        return f"{label} est très proche de la moyenne des vidéos performantes."
    elif diff > 0:
        return f"{label} est plus élevé que la moyenne. Essayez de raccourcir votre titre ou allonger votre description."
    else:
        return f"{label} est plus bas que la moyenne. Essayez de raccourcir votre description ou allonger votre titre."

SIM_CARD_MIN = 0.40  #minimum for anything below this is too different to show the otehr cards so hide them

def _maybe_text_card(field_label: str, sim: float):
    if sim >= SIM_CARD_MIN:
        return {"field": field_label, "similarity": sim, "message": _sim_message(sim)}
    return None

#realistic stuff using a dict
def classify_uniqueness_and_score(score: float, sims: list[float]) -> dict:

    very_diff_cut = 0.50    #<0.50 = very different
    moderately_sim_cut = 0.70  #0.50–0.70 = different but idk could be interesting

    low_count = sum(s < very_diff_cut for s in sims)            #how many are very different
    mid_count = sum((very_diff_cut <= s < moderately_sim_cut) for s in sims)

    #score tiers by using the percentiles from dataset
    def score_tier(s: float) -> str:
        if s >= 15.13:  #Q95
            return "viral"
        if s >= 12.92:  #Q75
            return "very_good"
        if s >= 11.00:  #Q50
            return "good"
        if s >= 9.00:   #Q25
            return "ok"
        if s >= 5.00:
            return "weak"
        return "poor"


    tier = score_tier(score)
    very_unique = (low_count >= 2)                #at least two fields very different
    somewhat_unique = (not very_unique) and (mid_count >= 2)

    #now i gotta craft the messgae
    if very_unique and tier in {"viral","very_good"}:
        title = "Originalité payante"
        msg = ("Votre contenu est original tout en obtenant un score élevé. "
               "Gardez cet angle différenciant, c’est un bon pari.")
    elif very_unique and tier in {"good","ok"}:
        title = "Original mais à affiner"
        msg = ("Votre contenu est assez unique. Le potentiel est correct, mais alignez un peu le titre/description "
               "avec des mots-clés plus proches des formats performants, sans perdre votre angle.")
    elif very_unique and tier in {"weak","poor"}:
        title = "Risque d’originalité"
        msg = ("Votre contenu est très différent et le score est bas. "
               "Renforcez la clarté des mots-clés et la promesse du titre pour aider l’algorithme à comprendre.")
    elif somewhat_unique and tier in {"viral","very_good","good"}:
        title = "Différenciation maîtrisée"
        msg = ("Votre contenu se distingue légèrement des standards tout en restant performant. "
               "Conservez cette touche originale.")
    elif somewhat_unique and tier in {"ok","weak"}:
        title = "Différenciation à cadrer"
        msg = ("Votre contenu sort un peu des codes. Ajoutez 1–2 mots-clés ‘classiques’ et un verbe d’action au titre "
               "pour gagner en clarté.")
    else:
        #meh not really unique
        if tier in {"viral","very_good"}:
            title = "Aligné et performant"
            msg = ("Votre contenu est bien aligné avec ce qui fonctionne actuellement, doublez la mise.")
        elif tier in {"good","ok"}:
            title = "Assez aligné, marge d’amélioration"
            msg = ("C’est dans le bon sens. Renforcez le titre (bénéfice clair, chiffre, verbe d’action) "
                   "et précisez 2–3 mots-clés fortement pertinents.")
        else:
            title = "Trop générique et peu performant"
            msg = ("Le contenu paraît générique pour l’algorithme. Spécifiez la promesse dans le titre, "
                   "allongez la description et utilisez des mots-clés concrets.")

    return {"field": title, "message": msg}


#create the feedback thatll be shown to user
def generate_feedback(video_type: str, numeric_values: dict, embeddings: dict, predicted_score: float):
    embeddings = embeddings or {}
    feedback = {"numeric": [], "text": []}

    if video_type == "long":
        feature_impacts = feature_impact_long
    else:
        feature_impacts = feature_impact_short

    #now gotta do the numeric cards
    for feat, val in numeric_values.items():
        if feat not in feature_impacts:
            continue
        avg_top = feature_impacts[feat]["avg_top10"]
        impact = feature_impacts[feat]["impact_score"]
        diff = float(val) - float(avg_top)
        label = FEATURE_LABELS.get(feat, feat)

        if video_type == "long":
            if feat == "duration_min":
                msg = _duration_message(label, diff)
            elif "ratio" in feat:
                msg = _ratio_message(label, diff)
            else:
                msg = _generic_message(label, diff, is_ratio=False)
        else:
            msg = _diff_message(label, diff, is_ratio=("ratio" in feat))

        feedback["numeric"].append({
            "feature": label,
            "value": float(val),
            "avg_top": float(avg_top),
            "impact_score": float(impact),
            "message": msg
        })

    feedback["numeric"].sort(key=lambda x: x["impact_score"], reverse=True)

    #and then gotta do text similarity 
    sims = []
    text_cards = []

    def _safe_sim(vec: np.ndarray | None, avg: np.ndarray | None) -> float | None:
        if vec is None or avg is None:
            return None
        return float(cosine_similarity([vec], [avg])[0][0])

    if video_type == "long":
        sim_title = _safe_sim(embeddings.get("title"),     avg_long_title)
        sim_desc  = _safe_sim(embeddings.get("description"), avg_long_desc)
        sim_kw    = _safe_sim(embeddings.get("keywords"),  avg_long_keywords)

        #use only available sims
        available = [s for s in (sim_title, sim_desc, sim_kw) if s is not None]
        if not available:
            available = [0.0]  #neutral fallback to prevent errors idk tho could remove it later
        combo = classify_uniqueness_and_score(predicted_score, available)
        text_cards.append({"field": combo["field"], "similarity": 0.0, "message": combo["message"]})

        if sim_title is not None:
            c = _maybe_text_card(TEXT_FIELD_LABELS["Title"], sim_title)
            if c: text_cards.append(c)
        if sim_desc is not None:
            c = _maybe_text_card(TEXT_FIELD_LABELS["Description"], sim_desc)
            if c: text_cards.append(c)
        if sim_kw is not None:
            c = _maybe_text_card(TEXT_FIELD_LABELS["Keywords"], sim_kw)
            if c: text_cards.append(c)

    else:  #short
        sim_title = _safe_sim(embeddings.get("title"),     avg_short_title)
        sim_desc  = _safe_sim(embeddings.get("description"), avg_short_desc)

        available = [s for s in (sim_title, sim_desc) if s is not None]
        if not available:
            available = [0.0]
        combo = classify_uniqueness_and_score(predicted_score, available)
        text_cards.append({"field": combo["field"], "similarity": 0.0, "message": combo["message"]})

        if sim_title is not None:
            c = _maybe_text_card(TEXT_FIELD_LABELS["Title"], sim_title)
            if c: text_cards.append(c)
        if sim_desc is not None:
            c = _maybe_text_card(TEXT_FIELD_LABELS["Description"], sim_desc)
            if c: text_cards.append(c)

    feedback["text"] = text_cards
    return feedback

#now for predict adn get this backend finished FINALLY
@app.post("/predict")
async def predict(
    video_type: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    keywords: Optional[str] = Form(""),
    file: UploadFile = File(...)
):
    emb_dict = {}

    #save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        #extract duration
        duration = get_video_duration(tmp_path)

        # starndardize inputs
        title_norm = normalise_text(title)
        desc_norm = normalise_text(description)
        keywords_norm = normalise_text(keywords) if keywords else ""

        #now to bild that class VideoData
        video_data = VideoData(
            video_type=video_type.lower(),
            title=title,
            description=description,
            keywords=keywords,
            duration=duration
        )

        #nwo the rest of feature engineering features + prediction + embeddings
        if video_data.video_type == "long":
            features = create_features_long(video_data)
            prediction = long_model.predict(features)[0]

            numeric_dict = {
                "duration_min": duration / 60,
                "desc_len": len(desc_norm)
            }
            emb_dict = {
                "title": _encode(title_norm),
                "description": _encode(desc_norm),
                "keywords": _encode(keywords_norm),
            }
        
        elif video_data.video_type == "short":
            features = create_features_short(video_data)
            prediction = short_model.predict(features)[0]

            numeric_dict = {
                "title_desc_len_ratio": len(title_norm) / max(1, len(desc_norm)),
                "desc_len": len(desc_norm),
                "title_stopword_ratio": sum(
                    w.lower() in {"the","and","a","to","of","in","is","you","that","it","on","for","with","as","at","by","an"}
                    for w in title_norm.split()
                ) / max(1, len(title_norm.split()))
            }
            emb_dict = {
                "title": _encode(title_norm),
                "description": _encode(desc_norm),
            }

        else:
            return {"error": "Invalid video_type. Use 'long' or 'short'."}

        #do feedback here
        predicted_score = float(prediction)
        feedback = generate_feedback(video_data.video_type, numeric_dict, emb_dict, predicted_score)

        #convert score to views
        predicted_views = np.expm1(prediction)


        #here we actually do the full response
        return {
            "video_type": video_data.video_type,
            "predicted_views": int(round(predicted_views)),
            "predicted_score": float(prediction),
            "duration_seconds": round(duration, 2),
            "feedback": feedback,
        }

    finally:
        #clean temp file even if something tfrg3at
        try:
            import os
            os.remove(tmp_path)
        except Exception:
            pass