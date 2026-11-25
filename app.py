"""Streamlit app: salary prediction for the Kazakhstan job market (hh.kz)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import (EXP_MAP, FEATURES, add_title_flags, build_matrix, clean_title, n_words, strip_html)

st.set_page_config(page_title="Прогноз зарплаты · Казахстан", page_icon="💼", layout="centered")

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"

EXP_LABELS = {"noExperience": "Без опыта", "between1And3": "1–3 года", "between3And6": "3–6 лет", "moreThan6": "Более 6 лет"}
SCHEDULE_LABELS = {"fullDay": "Полный день", "shift": "Сменный график", "flexible": "Гибкий график",
                   "remote": "Удалённая работа", "flyInFlyOut": "Вахтовый метод"}
EMPLOYMENT_LABELS = {"full": "Полная занятость", "part": "Частичная занятость", "project": "Проектная работа",
                     "probation": "Стажировка", "volunteer": "Волонтёрство"}


@st.cache_resource(show_spinner="Загружаю модель…")
def load_bundle():
    return joblib.load(MODEL_PATH)


def make_row(meta, title, area, role, experience, employment, schedule, gross_cat, skills, requirement, responsibility):
    caps = meta["word_caps"]
    coords = meta["area_coords"].get(area, meta["default_coords"])
    title_c = clean_title(title)
    req_c, resp_c = strip_html(requirement), strip_html(responsibility)
    skills_l = [s.strip().lower() for s in skills if s.strip()]
    # Empty text fields fall back to the typical (median) posting length seen in training
    med = dict(zip(["n_skills", "title_words", "desc_words", "req_words", "resp_words"],
                   meta["num_log_medians"]))
    req_w = n_words(req_c) or med["req_words"]
    resp_w = n_words(resp_c) or med["resp_words"]
    desc_w = max(req_w + resp_w, med["desc_words"])
    row = {
        "n_skills": len(skills_l), "title_words": n_words(title_c),
        "desc_words": min(desc_w, caps["desc_words"]),
        "req_words": min(req_w, caps["req_words"]),
        "resp_words": min(resp_w, caps["resp_words"]),
        "experience_ord": EXP_MAP[experience], "month_idx": meta["month_idx"],
        "latitude": coords["latitude"], "longitude": coords["longitude"],
        "has_coords": meta["has_coords_mode"], "has_address": meta["has_address_mode"],
        "has_range": meta["has_range_mode"],
        "area": area, "role_primary": role, "employment": employment, "schedule": schedule,
        "gross_cat": gross_cat, "currency": "KZT",
        "title_clean": title_c, "skills_clean": "|".join(skills_l),
    }
    df = add_title_flags(pd.DataFrame([row]))
    text = f"{title_c} {req_c} {resp_c} {req_c} {resp_c}".lower()
    return df[FEATURES], pd.Series([text])


def predict(bundle, X, texts, round_to=5000):
    B = build_matrix(bundle, X, texts)
    lo = np.expm1(bundle["q_lo"].predict(B) - bundle["qhat"])
    hi = np.expm1(bundle["q_hi"].predict(B) + bundle["qhat"])
    mid = np.clip(np.expm1(bundle["hgb"].predict(B)), lo, hi)
    rnd = lambda v: int(np.round(v[0] / round_to) * round_to)
    return rnd(lo), rnd(mid), rnd(hi)


def fmt_int(v):
    return f"{v:,.0f}".replace(",", "\u202f")


def fmt(v):
    return fmt_int(v) + " ₸"


bundle = load_bundle()
meta = bundle["meta"]
meta.setdefault("num_log_medians",
                bundle["preprocessor"].named_transformers_["num_log"].named_steps["imp"].statistics_.tolist())

st.title("💼 Прогноз зарплаты")
st.caption("Казахстан · вакансии hh.kz · Data Mining Final Project, KBTU")

with st.form("vacancy"):
    title = st.text_input("Название вакансии", "Python разработчик")
    c1, c2 = st.columns(2)
    areas = meta["areas"]
    area = c1.selectbox("Город", areas, index=areas.index("Алматы") if "Алматы" in areas else 0)
    experience = c2.selectbox("Опыт работы", list(EXP_LABELS), index=1, format_func=EXP_LABELS.get)
    roles = meta["roles"]
    role = st.selectbox("Профессиональная роль", roles,
                        index=roles.index("Программист, разработчик") if "Программист, разработчик" in roles else 0)
    c3, c4, c5 = st.columns(3)
    schedule = c3.selectbox("График", meta["schedule"], format_func=lambda s: SCHEDULE_LABELS.get(s, s))
    employment = c4.selectbox("Занятость", meta["employment"], format_func=lambda s: EMPLOYMENT_LABELS.get(s, s))
    gross_cat = c5.selectbox("Зарплата указана", ["net", "gross", "unknown"],
                             format_func={"net": "На руки", "gross": "До вычета налогов", "unknown": "Не указано"}.get)
    skills = st.multiselect("Ключевые навыки", meta["top_skills"], default=[s for s in ["python", "sql", "git"] if s in meta["top_skills"]],
                            accept_new_options=True)
    requirement = st.text_area("Требования (необязательно)", height=90,
                               placeholder="Опыт коммерческой разработки от 2 лет, знание Django/FastAPI…")
    responsibility = st.text_area("Обязанности (необязательно)", height=90,
                                  placeholder="Разработка backend-сервисов, code review…")
    submitted = st.form_submit_button("Рассчитать зарплату", type="primary", use_container_width=True)

if submitted:
    if not title.strip():
        st.warning("Введите название вакансии.")
    else:
        X, texts = make_row(meta, title, area, role, experience, employment, schedule, gross_cat, skills,
                            requirement, responsibility)
        lo, mid, hi = predict(bundle, X, texts)
        st.subheader("Результат")
        m1, m2, m3 = st.columns(3)
        m1.metric("Нижняя граница", fmt(lo))
        m2.metric("Ожидаемая зарплата", fmt(mid))
        m3.metric("Верхняя граница", fmt(hi))
        st.caption("Диапазон — 80% интервал: на тестовой выборке реальная зарплата попадает в него "
                   f"в {meta['test_metrics']['coverage_80_%']:.0f}% случаев.")

with st.expander("О модели"):
    tm = meta["test_metrics"]
    st.markdown(
        f"""
- **Данные:** HeadHunter Kazakhstan Jobs & Salary Dataset ({meta['period']}), {fmt_int(meta['n_rows'])} вакансий после очистки.
- **Модель:** HistGradientBoosting на структурных признаках + TF-IDF/SVD текста + стекинг текстовой Ridge-модели + target encoding;
  диапазон — квантильные модели с конформной калибровкой.
- **Качество на тесте:** R² (log) = {tm['R2_log']:.3f}, MAE = {fmt(tm['MAE_KZT'])}, медианная ошибка = {tm['median_APE_%']:.1f}%.
- Прогноз — ориентир по рынку, а не гарантия.
""")
