"""Retrain the deployable salary model from the hh.kz parquet file.

Reproduces the notebook pipeline (cleaning -> group-aware split -> preprocessing ->
TF-IDF text SVD + stacked text Ridge + target encoding -> tuned HistGradientBoosting
+ calibrated 80% quantile range). The sentence-embedding features are left out so the
app runs without PyTorch on free hosting.

Usage:  python train.py path/to/0000.parquet
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import (BIN_COLS, CAT_COLS, EXP_MAP, FEATURES, NUM_LOG, NUM_STD, TE_M, TEXT_SKILLS, TEXT_TITLE,
                      add_title_flags, build_matrix, clean_title, parse_roles, parse_skills, split_skills,
                      strip_html, te_apply, te_fit, te_keys)

SEED = 42
COVERAGE = 0.80
RATES_TO_KZT = {"KZT": 1.0, "RUR": 5.3, "RUB": 5.3, "USD": 500.0, "EUR": 545.0}
# Best parameters found by the extended search in the notebook (Section 8, round 2)
HGB2_PARAMS = {"min_samples_leaf": 10, "max_leaf_nodes": 127, "max_iter": 1000, "max_features": 0.5,
               "max_depth": 8, "max_bins": 255, "learning_rate": 0.02, "l2_regularization": 0}
TXT_ALPHA = 3.0  # best text-Ridge alpha from the notebook

OUT = Path(__file__).parent / "model" / "model.joblib"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_clean(path):
    df = pd.read_parquet(path)
    log(f"raw shape {df.shape}")
    df["period_dt"] = pd.to_datetime(df["period"].astype(str).str.strip(), format="%B %Y", errors="coerce")
    if df["period_dt"].isna().all():
        df["period_dt"] = pd.to_datetime(df["period"].astype(str), errors="coerce")

    content_cols = ["name", "area", "salary_from", "salary_to", "currency", "requirement", "responsibility"]
    df = df.drop_duplicates()
    df = df.sort_values("period_dt", kind="stable").drop_duplicates(subset="id", keep="first")
    df = df.drop_duplicates(subset=content_cols, keep="first")
    df = df.drop_duplicates(subset=[c for c in content_cols if c != "area"], keep="first")

    df["currency"] = df["currency"].astype("string").str.upper().str.strip()
    df["rate"] = df["currency"].map(RATES_TO_KZT).astype(float)
    df = df[df["rate"].notna()].copy()
    df["currency"] = df["currency"].astype(str)
    df["salary_from"] = pd.to_numeric(df["salary_from"], errors="coerce")
    df["salary_to"] = pd.to_numeric(df["salary_to"], errors="coerce")
    df["has_range"] = (df["salary_from"].notna() & df["salary_to"].notna()).astype(int)
    df["salary_kzt"] = df[["salary_from", "salary_to"]].mean(axis=1) * df["rate"]
    df = df[df["salary_kzt"] > 0].copy()
    df["gross_cat"] = df["gross"].astype("string").str.lower().map({"true": "gross", "false": "net"}).fillna("unknown").astype(str)

    df["has_address"] = df["full_address"].fillna("").astype(str).str.strip().ne("").astype(int)
    for c in ["name", "requirement", "responsibility", "description", "key_skills", "full_address"]:
        df[c] = df[c].fillna("").astype(str)
    for c in ["experience", "employment", "schedule", "area", "professional_roles"]:
        df[c] = df[c].fillna("Unknown").astype(str)
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df["latitude"].notna() & ~(df["latitude"].between(40, 56) & df["longitude"].between(46, 88))
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    df["has_coords"] = df["latitude"].notna().astype(int)
    for c in ["latitude", "longitude"]:
        df[c] = df[c].fillna(df.groupby("area")[c].transform("median")).fillna(df[c].median())
    if df["period_dt"].isna().any():
        df["period_dt"] = df["period_dt"].fillna(df["period_dt"].mode().iloc[0])

    df["log_salary"] = np.log1p(df["salary_kzt"])
    q1, q3 = df["log_salary"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[df["log_salary"].between(q1 - 3 * iqr, q3 + 3 * iqr)].copy()

    for c in ["description", "requirement", "responsibility"]:
        df[c + "_clean"] = df[c].map(strip_html)
    df["title_clean"] = df["name"].map(clean_title)
    df["skills_list"] = df["key_skills"].map(parse_skills)
    df["skills_clean"] = df["skills_list"].map(lambda L: "|".join(L))
    df["role_primary"] = df["professional_roles"].map(lambda v: parse_roles(v)[0])
    df["n_skills"] = df["skills_list"].str.len()
    df["title_words"] = df["title_clean"].str.split().str.len()
    df["desc_words"] = df["description_clean"].str.split().str.len()
    df["req_words"] = df["requirement_clean"].str.split().str.len()
    df["resp_words"] = df["responsibility_clean"].str.split().str.len()
    caps = {}
    for c in ["desc_words", "req_words", "resp_words"]:
        caps[c] = float(df[c].quantile(0.995))
        df[c] = df[c].clip(upper=caps[c])
    df["experience_ord"] = df["experience"].map(EXP_MAP)
    df["experience_ord"] = df["experience_ord"].fillna(df["experience_ord"].median())
    first = df["period_dt"].min()
    df["month_idx"] = (df["period_dt"].dt.year - first.year) * 12 + (df["period_dt"].dt.month - first.month)
    df = add_title_flags(df)
    df["text_full"] = (df["title_clean"] + " " + df["requirement_clean"] + " " + df["responsibility_clean"] + " "
                       + df["description_clean"]).str.lower()
    log(f"clean shape {df.shape}")
    return df, caps


def make_preprocessor():
    num_log_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("sc", StandardScaler())])
    num_std_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)
    title_pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_features=3000)),
                           ("svd", TruncatedSVD(30, random_state=SEED))])
    skills_pipe = Pipeline([("tfidf", TfidfVectorizer(tokenizer=split_skills, token_pattern=None, lowercase=False,
                                                      min_df=20, max_features=2000)),
                            ("svd", TruncatedSVD(30, random_state=SEED))])
    return ColumnTransformer([
        ("num_log", num_log_pipe, NUM_LOG), ("num_std", num_std_pipe, NUM_STD), ("bin", "passthrough", BIN_COLS),
        ("cat", cat_pipe, CAT_COLS), ("title", title_pipe, TEXT_TITLE), ("skills", skills_pipe, TEXT_SKILLS)])


def metrics(y_log, p_log):
    yt, yp = np.expm1(y_log), np.expm1(p_log)
    return {"R2_log": float(r2_score(y_log, p_log)), "MAE_KZT": float(mean_absolute_error(yt, yp)),
            "MAPE_%": float(np.mean(np.abs(yt - yp) / yt) * 100),
            "median_APE_%": float(np.median(np.abs(yp / yt - 1)) * 100)}


def main(path):
    df, caps = load_and_clean(path)
    X, y = df[FEATURES].copy(), df["log_salary"]

    key = (df["title_clean"] + "|" + df["salary_from"].fillna(-1).astype("int64").astype(str) + "|"
           + df["salary_to"].fillna(-1).astype("int64").astype(str) + "|" + df["currency"].astype(str))
    groups = pd.factorize(key)[0]
    strata = pd.qcut(y, 10, labels=False, duplicates="drop")
    fold = np.zeros(len(X), dtype=int)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=SEED).split(X, strata, groups)):
        fold[idx] = k
    te_pos, va_pos, tr_pos = np.where(fold < 3)[0], np.where((fold >= 3) & (fold < 6))[0], np.where(fold >= 6)[0]
    X_tr, X_va, X_te = X.iloc[tr_pos], X.iloc[va_pos], X.iloc[te_pos]
    ytr, yva, yte = y.values[tr_pos], y.values[va_pos], y.values[te_pos]
    g_tr = groups[tr_pos]
    txt = df["text_full"]
    t_tr, t_va, t_te = txt.iloc[tr_pos], txt.iloc[va_pos], txt.iloc[te_pos]
    log(f"split train {len(tr_pos)} / val {len(va_pos)} / test {len(te_pos)}")

    pre = make_preprocessor().fit(X_tr)
    Xtr = pre.transform(X_tr).astype(np.float32)
    log("preprocessor fitted")

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=30, max_features=30000, sublinear_tf=True, dtype=np.float32)
    T_tr = tfidf.fit_transform(t_tr)
    svd = TruncatedSVD(60, random_state=SEED).fit(T_tr)
    ridge_txt = Ridge(alpha=TXT_ALPHA, solver="sparse_cg").fit(T_tr, ytr)
    P_tr = np.zeros(len(ytr), dtype=np.float32)
    for a, b in GroupKFold(5).split(T_tr, ytr, g_tr):
        P_tr[b] = Ridge(alpha=TXT_ALPHA, solver="sparse_cg").fit(T_tr[a], ytr[a]).predict(T_tr[b])
    log("text models fitted")

    ktr = te_keys(X_tr)
    prior = float(ytr.mean())
    TE_tr = np.zeros((len(ytr), len(TE_M)), dtype=np.float32)
    for a, b in GroupKFold(5).split(Xtr, ytr, g_tr):
        for j, n in enumerate(TE_M):
            TE_tr[b, j] = te_apply(ktr[n][b], te_fit(ktr[n][a], ytr[a]), ytr[a].mean(), TE_M[n])
    te_stats = {n: te_fit(ktr[n], ytr) for n in TE_M}

    tfidf.stop_words_ = None  # large and not needed for transform
    bundle = {"preprocessor": pre, "tfidf": tfidf, "svd": svd, "ridge_txt": ridge_txt,
              "te_stats": te_stats, "te_prior": prior}

    # Training matrix uses out-of-fold stacked/target-encoded columns (as in the notebook)
    B_tr = np.hstack([Xtr, svd.transform(T_tr).astype(np.float32), P_tr.reshape(-1, 1), TE_tr]).astype(np.float32)
    B_va = build_matrix(bundle, X_va, t_va)
    B_te = build_matrix(bundle, X_te, t_te)

    common = dict(early_stopping=True, n_iter_no_change=20, random_state=SEED, **HGB2_PARAMS)
    hgb = HistGradientBoostingRegressor(**common).fit(B_tr, ytr)
    log(f"HGB fitted ({hgb.n_iter_} iterations)")
    lo_q, hi_q = (1 - COVERAGE) / 2, 1 - (1 - COVERAGE) / 2
    q_lo = HistGradientBoostingRegressor(loss="quantile", quantile=lo_q, **common).fit(B_tr, ytr)
    q_hi = HistGradientBoostingRegressor(loss="quantile", quantile=hi_q, **common).fit(B_tr, ytr)
    scores = np.maximum(q_lo.predict(B_va) - yva, yva - q_hi.predict(B_va))
    qhat = float(np.quantile(scores, min(1.0, np.ceil((len(yva) + 1) * COVERAGE) / len(yva))))
    log("quantile models fitted")

    p_te = hgb.predict(B_te)
    lo_te, hi_te = q_lo.predict(B_te) - qhat, q_hi.predict(B_te) + qhat
    test = {**metrics(yte, p_te), "coverage_80_%": float(np.mean((yte >= lo_te) & (yte <= hi_te)) * 100),
            "n_test": int(len(yte))}
    log(f"TEST: {test}")

    full = df  # dictionaries for the UI come from all clean data
    area_counts = full["area"].value_counts()
    role_counts = full["role_primary"].value_counts()
    skill_counts = full["skills_list"].explode().dropna().value_counts()
    meta = {
        "areas": area_counts[area_counts >= 30].index.tolist(),
        "roles": role_counts[role_counts >= 30].index.tolist(),
        "employment": full["employment"].value_counts().index.tolist(),
        "schedule": full["schedule"].value_counts().index.tolist(),
        "top_skills": skill_counts.head(400).index.tolist(),
        "area_coords": full.groupby("area")[["latitude", "longitude"]].median().to_dict("index"),
        "default_coords": {"latitude": float(full["latitude"].median()), "longitude": float(full["longitude"].median())},
        "month_idx": int(full["month_idx"].max()),
        "has_coords_mode": int(full["has_coords"].mode()[0]),
        "has_address_mode": int(full["has_address"].mode()[0]),
        "has_range_mode": int(full["has_range"].mode()[0]),
        "word_caps": caps,
        "test_metrics": test,
        "n_rows": int(len(full)),
        "period": f"{full['period_dt'].min():%Y-%m} — {full['period_dt'].max():%Y-%m}",
    }
    bundle.update({"hgb": hgb, "q_lo": q_lo, "q_hi": q_hi, "qhat": qhat, "meta": meta})
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT, compress=3)
    log(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0000.parquet")
