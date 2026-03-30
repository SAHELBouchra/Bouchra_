import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def merge_excel(input_excel_path, output_csv_path):
    xls = pd.ExcelFile(input_excel_path)
    all_dfs = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        print(f"✅ Feuille chargée : {sheet_name} → {len(df)} lignes")
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)

    long_df = df.rename(columns={
        "UserID": "user_id", "Sexe": "sex", "Âge": "age", "Taille": "height_cm",
        "Poids": "weight_kg", "Sport": "Sport_type", "PA": "activity_freq",
        "Sommeil": "sleep_duration", "Stress": "stress_level", "Nutri": "nutrition_raw",
        "Sédentaire": "sedentary_time", "Alcool": "alcohol_raw",
        "Antécédents": "family_history_raw", "Cycle": "cycle_raw",
        "Date": "date", "Heure": "time", "Phase": "phase", "Status": "status"
    }).copy()

    long_df["datetime"] = pd.to_datetime(
        long_df["date"].astype(str) + " " + long_df["time"].astype(str),
        format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )
    long_df = long_df.sort_values(["user_id", "datetime"])

    static_cols = ["sex", "age", "height_cm", "weight_kg"]
    for col in static_cols:
        long_df[col] = long_df.groupby("user_id")[col].ffill().bfill().infer_objects(copy=False)

    long_df.to_csv(output_csv_path, index=False)
    print(f"✅ Merge terminé – {len(long_df)} lignes totales")
    return output_csv_path


def clean_mapping(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    def bin_sleep(x):
        if pd.isna(x): return None
        if x >= 8: return ">8h"
        elif x >= 7: return "7–8h"
        elif x >= 6: return "6–7h"
        else: return "<6h"

    def bin_stress(x):
        if pd.isna(x): return None
        if x >= 7: return "high"
        elif x >= 4: return "moderate"
        else: return "low"

    def bin_activity(x):
        if pd.isna(x): return None
        if x >= 5: return "≥5"
        elif x >= 4: return "4"
        elif x >= 3: return "3"
        elif x >= 2: return "2"
        else: return "0–1"

    df["sleep_duration"] = df["sleep_duration"].apply(bin_sleep)
    df["stress_level"] = df["stress_level"].apply(bin_stress)
    df["activity_freq"] = df["activity_freq"].apply(bin_activity)

    df["sex"] = df["sex"].astype(str).str.upper().map({"F": "female", "H": "male", "M": "male"})
    df["sex"] = df.groupby('user_id')['sex'].ffill()

    df["height_m"] = df["height_cm"] / 100
    df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)

    stress_map = {"Faible": "low", "Modéré": "moderate", "Élevé": "high"}
    df["stress_level_norm"] = df["stress_level"].map(stress_map).fillna("moderate")

    df["sleep_6h_plus_norm"] = df["sleep_duration"].apply(lambda x: 0 if str(x).strip() == "<6h" else 1).fillna(1)

    nutrition_map = {"Maison": "equilibrated", "Mix maison": "mixed", "Mix": "mixed", 
                     "Équilibrée": "equilibrated", "Standard": "mixed"}
    df["nutrition_norm"] = df["nutrition_raw"].map(nutrition_map).fillna("mixed")

    df["family_history_flag"] = df["family_history_raw"].apply(lambda x: 0 if x == "Aucun" else 1)

    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def calculate_scores(input_csv_path, output_scores_csv_path, output_features_csv_path):
    df = pd.read_csv(input_csv_path)

    # Body Age
    activity_score_map_body_age = {"0–1": 3, "0": 3, "1": 3, "2–3": 1, "2": 1, "3": 1, 
                                   "4": 0, "4–5": 0, "5": 0, "≥5": -2, "≥6": -2}
    def bmi_score(bmi):
        if bmi < 18.5: return 1
        elif 18.5 <= bmi < 25: return 0
        elif 25 <= bmi < 30: return 1
        else: return 3

    sleep_score_map_body_age = {"<6h": 1, "6–7h": 0, "7–8h": 0, ">8h": -1}
    stress_score_map_body_age = {"high": 2, "moderate": 1, "low": 0}
    nutrition_score_map_body_age = {"poor": 2, "mixed": 0, "equilibrated": -1}

    df["activity_score"] = df["activity_freq"].map(activity_score_map_body_age).fillna(0)
    df["bmi_score"] = df["bmi"].apply(bmi_score)
    df["sleep_score"] = df["sleep_duration"].map(sleep_score_map_body_age).fillna(0)
    df["stress_score"] = df["stress_level_norm"].map(stress_score_map_body_age).fillna(0)
    df["nutrition_score"] = df["nutrition_norm"].map(nutrition_score_map_body_age).fillna(0)

    df["body_age"] = df["age"] + df["activity_score"] + df["bmi_score"] + df["sleep_score"] + df["stress_score"] + df["nutrition_score"]

    # Work Load
    activity_freq_map_workload = {"0–1": 0, "0": 0, "1": 0, "2–3": 10, "2": 10, "3": 10, 
                                  "4": 20, "4–5": 20, "5": 20, "≥5": 20, "≥6": 30}
    stress_map_workload = {"low": 0, "moderate": -10, "high": -20}
    df["intensity_score_workload"] = 0
    df["activity_score_workload"] = df["activity_freq"].map(activity_freq_map_workload).fillna(0)
    df["stress_score_workload"] = df["stress_level_norm"].map(stress_map_workload).fillna(0)
    df["sleep_debt_score_workload"] = df["sleep_6h_plus_norm"].apply(lambda x: 0 if x == 1 else 20)
    df["work_load"] = df["intensity_score_workload"] + df["activity_score_workload"] - df["stress_score_workload"] - df["sleep_debt_score_workload"]

    # Body Toxins
    nutrition_exposure_map_toxins = {"ultra_processed": 2, "mixed": 0, "equilibrated": 0}
    alcohol_exposure_map_toxins = {"Jamais": 0, "Occasionnel": 0, "Régulier": 2, "1–3/sem": 2, ">3/sem": 2}
    df["Hydratation_score_toxins"] = 0
    df["nutrition_toxin"] = df["nutrition_norm"].map(nutrition_exposure_map_toxins).fillna(0)
    df["alcohol_toxin"] = df["alcohol_raw"].map(alcohol_exposure_map_toxins).fillna(0)
    df["sport_toxins"] = df["activity_freq"].apply(lambda x: -2 if x in ["3", "4", "4–5", "≥5"] else 0)
    df["Body_Toxins"] = df["nutrition_toxin"] + df["alcohol_toxin"] - (df["Hydratation_score_toxins"] + df["sport_toxins"])

    final_features = df[["user_id", "datetime", "sex", "age", "height_cm", "weight_kg", "bmi",
                         "phase", "status", "Sport_type", "activity_freq", "sleep_duration",
                         "stress_level", "nutrition_raw", "sedentary_time", "alcohol_raw",
                         "family_history_raw", "cycle_raw"]].copy()

    final_features["body_age"] = df["body_age"]
    final_features["work_load"] = df["work_load"]
    final_features["body_toxin"] = df["Body_Toxins"]

    # AJOUT DES COLONNES NORMALISÉES (IMPORTANT)
    final_features["stress_level_norm"] = df["stress_level_norm"]
    final_features["sleep_6h_plus_norm"] = df["sleep_6h_plus_norm"]
    final_features["nutrition_norm"] = df["nutrition_norm"]
    final_features["family_history_flag"] = df["family_history_flag"]

    final_features.to_csv(output_features_csv_path, index=False)
    final_features.to_csv(output_scores_csv_path, index=False)
    return output_scores_csv_path


def calculate_body_systems(df):
    def norm(val, minv=0, maxv=10):
        return np.clip(((val - minv) / (maxv - minv) * 10), 0, 10)

    # Fallbacks sécurisés
    if 'stress_level_norm' not in df.columns: df['stress_level_norm'] = "moderate"
    if 'sleep_6h_plus_norm' not in df.columns: df['sleep_6h_plus_norm'] = 1
    if 'nutrition_norm' not in df.columns: df['nutrition_norm'] = "mixed"
    if 'sedentary_time' not in df.columns: df['sedentary_time'] = 4
    if 'body_age_change' not in df.columns: df['body_age_change'] = 0
    if 'work_load_change' not in df.columns: df['work_load_change'] = 0
    if 'body_toxin_change' not in df.columns: df['body_toxin_change'] = 0

    df["inflammation_score"] = (
        norm(df["stress_level_norm"].map({"low":0,"moderate":5,"high":10})) +
        norm(10 - df["sleep_6h_plus_norm"]*10) +
        norm(df["body_toxin"]*5)
    ) / 3

    df["metabolic_score"] = (
        norm(df["bmi"], 18.5, 35) +
        norm(df["sedentary_time"]) +
        norm(df["nutrition_norm"].map({"equilibrated":0,"mixed":5,"poor":10}))
    ) / 3

    df["stress_hormonal_score"] = (
        norm(df["stress_level_norm"].map({"low":0,"moderate":5,"high":10})) +
        norm(df["work_load"], 0, 50) +
        norm(10 - df["sleep_6h_plus_norm"]*10)
    ) / 3

    df["muscular_recovery_score"] = (
        norm(df["activity_freq"].map({"0–1":0,"2":3,"3":5,"4":7,"≥5":10})) +
        norm(10 - df["sleep_6h_plus_norm"]*10) +
        norm(df["work_load"], 0, 50)
    ) / 3

    df["liver_detox_score"] = (
        norm(df["body_toxin"], 0, 10) +
        norm(df["alcohol_raw"].map({"Jamais":0,"Occasionnel":3,"1–3/sem":6,"Régulier":10})) +
        norm(df["nutrition_norm"].map({"equilibrated":0,"mixed":5,"poor":10}))
    ) / 3

    df["cardio_score"] = (
        norm(df["work_load"], 0, 50) +
        norm(df["sedentary_time"]) +
        norm(df["stress_level_norm"].map({"low":0,"moderate":5,"high":10}))
    ) / 3

    df["toxins_oxidative_score"] = (
        norm(df["body_toxin"], 0, 10) +
        norm(df["alcohol_raw"].map({"Jamais":0,"Occasionnel":3,"1–3/sem":6,"Régulier":10})) +
        norm(df["sedentary_time"])
    ) / 3

    return df


def add_ml_predictions(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df = calculate_body_systems(df)

    features = ["age", "bmi", "body_age", "work_load", "body_toxin",
                "inflammation_score", "metabolic_score", "stress_hormonal_score",
                "muscular_recovery_score", "liver_detox_score", "cardio_score",
                "toxins_oxidative_score", "sleep_6h_plus_norm", "family_history_flag"]

    X = df[features].fillna(0)

    # 1. Injury Risk
    df["injury_proxy"] = ((df["work_load"] > 25) | (df["sleep_6h_plus_norm"] == 0) | 
                          (df["stress_level_norm"] == "high") | (df["inflammation_score"] > 6)).astype(int)
    model_inj = RandomForestClassifier(n_estimators=100, random_state=42)
    model_inj.fit(X, df["injury_proxy"])
    df["predicted_injury_risk_%"] = model_inj.predict_proba(X)[:, 1] * 100

    # 2. Chronic Risk
    df["chronic_proxy"] = ((df["bmi"] > 27) | (df["body_toxin"] > 3) | 
                           (df["metabolic_score"] > 6) | (df["family_history_flag"] == 1)).astype(int)
    model_chr = RandomForestClassifier(n_estimators=100, random_state=42)
    model_chr.fit(X, df["chronic_proxy"])
    df["predicted_chronic_risk_%"] = model_chr.predict_proba(X)[:, 1] * 100

    # 3. Bio Age 3 mois
    df["bio_age_future"] = df["body_age"] + df.get("body_age_change", 0) * 1.8
    model_bio = RandomForestRegressor(n_estimators=100, random_state=42)
    model_bio.fit(X, df["bio_age_future"])
    df["predicted_body_age_3m"] = model_bio.predict(X)
    df["bio_age_improvement"] = df["body_age"] - df["predicted_body_age_3m"]   # ← LIGNE AJOUTÉE ICI

    # 4. Performance
    df["perf_proxy"] = 100 - df["work_load"]*1.5 + (100 - df["body_age"]) + df["cardio_score"]*2
    model_perf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_perf.fit(X, df["perf_proxy"])
    df["predicted_performance_improvement_%"] = model_perf.predict(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_inj, "models/injury_model.pkl")
    joblib.dump(model_chr, "models/chronic_model.pkl")
    joblib.dump(model_bio, "models/bio_model.pkl")
    joblib.dump(model_perf, "models/perf_model.pkl")

    df.to_csv(output_csv_path, index=False)
    print("✅ ML terminé → final_with_predictions.csv créé")
    return output_csv_path


def digital_twin(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["user_id", "datetime"])

    df["delta_body_age"] = df["body_age"] - df["age"]
    def body_age_state(delta):
        if delta <= -2: return "younger_than_chrono"
        elif -1 <= delta <= 1: return "aligned_with_chrono"
        else: return "accelerated_aging"
    df["body_age_state"] = df["delta_body_age"].apply(body_age_state)
    df["body_age_change"] = df.groupby("user_id")["body_age"].diff().fillna(0)

    def workload_state(score):
        if score <= 0: return "low_load"
        elif score <= 30: return "moderate_load"
        else: return "high_load"
    df["workload_state"] = df["work_load"].apply(workload_state)
    df["work_load_change"] = df.groupby("user_id")["work_load"].diff().fillna(0)

    def toxin_state(score):
        if score <= 0: return "low_toxic_load"
        elif score <= 2: return "moderate_toxic_load"
        else: return "high_toxic_load"
    df["body_toxins_state"] = df["body_toxin"].apply(toxin_state)
    df["body_toxin_change"] = df.groupby("user_id")["body_toxin"].diff().fillna(0)

    final_cols = [
        "user_id", "datetime", "sex", "age", "height_cm", "weight_kg", "bmi",
        "phase", "status", "Sport_type", "activity_freq", "sleep_duration",
        "stress_level", "nutrition_raw", "sedentary_time", "alcohol_raw",
        "family_history_raw", "cycle_raw",
        "body_age", "body_age_change", "body_age_state",
        "work_load", "work_load_change", "workload_state",
        "body_toxin", "body_toxin_change", "body_toxins_state",
        "stress_level_norm", "sleep_6h_plus_norm", "nutrition_norm", "family_history_flag"
    ]

    df_final = df[final_cols].copy()
    df_final.to_csv(output_csv_path, index=False)
    return output_csv_path


def run_pipeline(input_excel):
    csv1 = 'temp_merged.csv'
    merge_excel(input_excel, csv1)
    
    csv2 = 'temp_cleaned.csv'
    clean_mapping(csv1, csv2)
    
    csv3 = 'temp_scores.csv'
    features_temp = 'temp_features.csv'
    calculate_scores(csv2, csv3, features_temp)
    
    final_csv = 'final_digital_twin.csv'
    digital_twin(csv3, final_csv)
    
    ml_csv = 'final_with_predictions.csv'
    add_ml_predictions(final_csv, ml_csv)
    
    print(f"✅ Pipeline + ML COMPLET ! Fichier final : {ml_csv}")
    
    for f in [csv1, csv2, csv3, features_temp]:
        if os.path.exists(f): os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    run_pipeline(args.input)