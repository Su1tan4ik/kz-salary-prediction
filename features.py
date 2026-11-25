"""Shared feature engineering for training and the web app (mirrors the notebook)."""
import ast
import html
import json
import re

import numpy as np
import pandas as pd

NUM_LOG = ["n_skills", "title_words", "desc_words", "req_words", "resp_words"]
NUM_STD = ["experience_ord", "month_idx", "latitude", "longitude", "title_level", "req_share"]
BIN_COLS = ["has_coords", "has_address", "has_range", "is_big_city", "is_oil_region", "is_remote", "is_it_title"]
CAT_COLS = ["area", "role_primary", "employment", "schedule", "gross_cat", "currency"]
TEXT_TITLE = "title_clean"
TEXT_SKILLS = "skills_clean"
FEATURES = NUM_LOG + NUM_STD + BIN_COLS + CAT_COLS + [TEXT_TITLE, TEXT_SKILLS]

EXP_MAP = {"noExperience": 0, "between1And3": 1, "between3And6": 2, "moreThan6": 3}

LEADERSHIP_KW = r"руковод|начальник|директор|заведующ|главн|head|director|chief|менеджер проект|управляющ"
SENIOR_KW = r"senior|старш|ведущ|lead|сеньор|опытн"
JUNIOR_KW = r"junior|младш|стажер|стажёр|помощник|ассистент|intern|trainee|начинающ"
IT_KW = r"разработчик|программист|developer|python|java|devops|qa|тестировщик|data|аналитик данных|software|инженер-программист|frontend|backend"

TE_M = {"te_title": 20, "te_city_role": 10, "te_role": 10}


def strip_html(s):
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_roles(v):
    s = str(v)
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            obj = [obj]
        if isinstance(obj, (list, tuple)):
            names = [o.get("name") if isinstance(o, dict) else str(o) for o in obj]
            names = [n for n in names if n]
            if names:
                return names
    return [s.strip() or "Unknown"]


def parse_skills(v):
    s = re.sub(r"[\[\]'\"]", "", str(v))
    return [p.strip().lower() for p in re.split(r"[,;\n|]", s) if p.strip()]


def clean_title(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def split_skills(s):
    return [t for t in s.split("|") if t]


def n_words(s):
    return len(str(s).split())


def add_title_flags(df):
    title = df["title_clean"]
    df["title_level"] = np.select(
        [title.str.contains(LEADERSHIP_KW, regex=True), title.str.contains(SENIOR_KW, regex=True),
         title.str.contains(JUNIOR_KW, regex=True)],
        [2, 1, -1], default=0)
    area = df["area"].str.lower()
    df["is_big_city"] = area.str.contains(r"алматы|астана|нур-султан|almaty|astana", regex=True).astype(int)
    df["is_oil_region"] = area.str.contains(r"атырау|актау|мангистау|жанаозен|atyrau|aktau", regex=True).astype(int)
    df["is_remote"] = (df["schedule"].str.lower() == "remote").astype(int)
    df["is_it_title"] = title.str.contains(IT_KW, regex=True).astype(int)
    df["req_share"] = df["req_words"] / (df["desc_words"] + 1)
    return df


def te_keys(frame):
    return {
        "te_title": frame["title_clean"].astype(str).to_numpy(),
        "te_city_role": (frame["area"].astype(str) + "|" + frame["role_primary"].astype(str)).to_numpy(),
        "te_role": frame["role_primary"].astype(str).to_numpy(),
    }


def te_fit(keys, y):
    return pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])


def te_apply(keys, stats, prior, m):
    s = stats.reindex(keys)
    return ((s["sum"].fillna(0).to_numpy() + m * prior) / (s["count"].fillna(0).to_numpy() + m)).astype(np.float32)


def build_matrix(bundle, X_rows, texts):
    """Full feature matrix used by the deployed gradient-boosting models."""
    base = bundle["preprocessor"].transform(X_rows).astype(np.float32)
    T = bundle["tfidf"].transform(texts)
    S = bundle["svd"].transform(T).astype(np.float32)
    P = bundle["ridge_txt"].predict(T).astype(np.float32).reshape(-1, 1)
    keys = te_keys(X_rows)
    TE = np.column_stack([te_apply(keys[n], bundle["te_stats"][n], bundle["te_prior"], TE_M[n]) for n in TE_M])
    return np.hstack([base, S, P, TE]).astype(np.float32)
