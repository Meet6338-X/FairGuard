"""Download open, license-friendly datasets from trusted public sources.

    python scripts/download_datasets.py

Sources (no account or API key needed):
- UCI Machine Learning Repository (freely donatable, citation requested)
  * Adult / Census income  — income_gt_50k label, sensitive: sex (also race, age)
  * Statlog German credit  — credit_good label, sensitive: sex
  * Default of credit card clients (Taiwan) — default_next_month label, sensitive: gender
- ProPublica COMPAS recidivism (open data journalism, CC BY)
  * two_year_recid label, sensitive: race (also sex)

Files land in data/ (git-ignored, regenerate any time with this script);
original downloads are kept in data/raw/. Personal-name columns are dropped
so the converted files carry no personal identifiers.
"""
import csv
import io
import zipfile
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import fairguard`
from fairguard import config, data

RAW = config.DATA_DIR / "raw"
SOURCES = {
    "adult.zip": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "german.zip": "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
    "taiwan.zip": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "compas.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    "bank.zip": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "student.zip": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "heart.zip": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    "communities.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
    "australian.dat": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat",
}

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]

GERMAN_COLS = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status", "other_parties", "residence_since", "property", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "target",
]
GERMAN_SEX = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}

COMPAS_KEEP = [
    "id", "age", "age_cat", "sex", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid",
]

SPECS = [
    # (output csv, label, sensitive) — printed as a usage table at the end
    ("adult.csv", "income_gt_50k", "sex"),
    ("german.csv", "credit_good", "sex"),
    ("taiwan_credit.csv", "default_next_month", "gender"),
    ("compas_two_years.csv", "two_year_recid", "race"),
    ("bank_marketing.csv", "subscribed", "age"),
    ("student_performance.csv", "G3_pass", "sex"),
    ("heart.csv", "disease", "sex"),
    ("communities.csv", "high_violent_crime", "majority_white"),
    ("australian.csv", "credit_approved", "gender_like"),
]


def fetch(name: str, url: str) -> Path:
    """Download url into data/raw/<name> once (skip when already there)."""
    dest = RAW / name
    if dest.exists():
        print(f"  keep    {name} (already downloaded)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "FairGuard/0.1 (academic project)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
    print(f"  fetched {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def unzip(zip_path: Path, member: str) -> str:
    """Return the text of one member inside a downloaded zip."""
    with zipfile.ZipFile(zip_path) as z:
        real = next(n for n in z.namelist() if n.lower().endswith(member.lower()) or n.lower() == member.lower())
        return z.read(real).decode("utf-8", errors="replace")


def unzip_bytes(zip_path: Path, member: str) -> bytes:
    """Return the raw bytes of one member inside a downloaded zip."""
    with zipfile.ZipFile(zip_path) as z:
        real = next(n for n in z.namelist() if n.lower().endswith(member.lower()) or n.lower() == member.lower())
        return z.read(real)


def convert_adult() -> pd.DataFrame:
    frames = []
    for member in ("adult.data", "adult.test"):
        text = unzip(RAW / "adult.zip", member)
        rows = []
        for line in text.splitlines():
            if not line.strip() or line.startswith("|"):
                continue
            rows.append([v.strip().rstrip(".") for v in line.split(",")])
        frames.append(pd.DataFrame(rows, columns=ADULT_COLS))
    df = pd.concat(frames, ignore_index=True)
    df["income_gt_50k"] = (df["income"] == ">50K").astype(int)
    numeric = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col])
    return df


def convert_german() -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(unzip(RAW / "german.zip", "german.data")), delimiter=" "))
    df = pd.DataFrame(rows, columns=GERMAN_COLS)
    df["sex"] = df["personal_status"].map(GERMAN_SEX)
    df["credit_good"] = (df["target"] == "1").astype(int)
    numeric = ["duration_months", "credit_amount", "installment_rate", "residence_since",
               "age", "existing_credits", "num_dependents"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c])
    return df.drop(columns=["target"])


def convert_taiwan_credit() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "taiwan.zip") as z:
        xlsx = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls")))
        raw = config.ROOT / "data" / "raw" / "taiwan_tmp.xlsx"
        raw.write_bytes(z.read(xlsx))
    df = pd.read_excel(raw, header=1, engine="xlrd")
    raw.unlink(missing_ok=True)
    df.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]
    df = df.rename(columns={"default_payment_next_month": "default_next_month"})
    df["gender"] = df["sex"].map({1: "male", 2: "female"})
    return df.drop(columns=["sex", "id"])


def convert_bank_marketing() -> pd.DataFrame:
    """UCI id 222 — 45k bank marketing records; 'y' -> subscribed.

    The 'duration' column is dropped on purpose: it leaks the target (call
    length is only known after the call), a documented trap of this dataset.
    """
    inner = zipfile.ZipFile(io.BytesIO(unzip_bytes(RAW / "bank.zip", "bank.zip")))
    text = inner.read("bank-full.csv").decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";")
    df["subscribed"] = (df["y"] == "yes").astype(int)
    return df.drop(columns=["y", "duration"])


def convert_student_performance() -> pd.DataFrame:
    """UCI id 320 — math + Portuguese student grades; G3 >= 10 -> passed.

    The final mark G3 is removed (it *is* the outcome); the two earlier period
    grades G1/G2 stay — they are legitimately available at prediction time.
    """
    inner = zipfile.ZipFile(io.BytesIO(unzip_bytes(RAW / "student.zip", "student.zip")))
    frames = []
    for member, course in (("student-mat.csv", "math"), ("student-por.csv", "people")):
        df = pd.read_csv(io.StringIO(inner.read(member).decode("utf-8")), sep=";")
        df["course"] = course
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["G3_pass"] = (df["G3"] >= 10).astype(int)
    return df.drop(columns=["G3"])


HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

COMMUNITIES_COLS = [
    "state", "county", "community", "communityname", "fold", "population",
    "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
    "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
    "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
    "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
    "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
    "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov",
    "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
    "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu",
    "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
    "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
    "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom",
    "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",
    "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
    "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
    "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous",
    "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup",
    "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant",
    "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
    "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
    "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ",
    "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc",
    "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn",
    "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85",
    "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
    "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
    "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
    "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
    "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked",
    "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",
    "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
    "PolicBudgPerPop", "ViolentCrimesPerPop",
]


def convert_heart() -> pd.DataFrame:
    """UCI id 45 (Cleveland heart) — 'value > 0' -> disease (narrowed to 0/1).

    '?' markers become NaN; the model median-imputes them, so no rows are lost.
    """
    df = pd.read_csv(io.StringIO(unzip(RAW / "heart.zip", "processed.cleveland.data")), header=None, names=HEART_COLS)
    df = df.replace("?", pd.NA)
    numeric = [c for c in HEART_COLS if c not in ("num",)]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["disease"] = (df["num"] > 0).fillna(False).astype(int)
    df["sex"] = df["sex"].map({1.0: "male", 0.0: "female"})
    return df.drop(columns=["num"])


def convert_communities() -> pd.DataFrame:
    """UCI Communities and Crime — 1994 US communities, socio-economic +
    law-enforcement attributes.

    Target: high_violent_crime = ViolentCrimesPerPop above the median.
    Sensitive: racePctWhite above median -> 'majority_white' (binary).
    Rows with missing target ('?') are dropped; remaining '?' become NaN.
    """
    text = (RAW / "communities.data").read_text(encoding="utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), header=None, names=COMMUNITIES_COLS)
    df = df.drop(columns=["state", "county", "community", "communityname", "fold"])
    df = df.replace("?", pd.NA)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop columns that are mostly missing (>50%) — they can't be features.
    missing_frac = df.isna().mean()
    df = df.loc[:, missing_frac <= 0.5]
    df = df.dropna(subset=["ViolentCrimesPerPop"])
    med = df["ViolentCrimesPerPop"].median()
    df["high_violent_crime"] = (df["ViolentCrimesPerPop"] > med).astype(int)
    med_white = df["racePctWhite"].median()
    df["majority_white"] = (df["racePctWhite"] > med_white).astype(int)
    return df.drop(columns=["ViolentCrimesPerPop", "racePctWhite"])


def convert_australian() -> pd.DataFrame:
    """UCI Statlog Australian Credit Approval — 690 credit card applications,
    14 attributes + class (attribute names anonymized A1..A15).

    Target: credit_approved = class '+'.
    Sensitive: gender-like attribute A9 (t/f).
    Missing values marked '?' become NaN.
    """
    text = (RAW / "australian.dat").read_text(encoding="utf-8", errors="replace")
    rows = [v.split() for v in text.splitlines() if v.strip()]
    cols = [f"A{i}" for i in range(1, 16)]
    df = pd.DataFrame(rows, columns=cols)
    df = df.replace("?", pd.NA)
    for col in cols[:14]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["credit_approved"] = (df["A15"] == "1").astype(int)
    # A9 is already 0/1 (gender-like); use it directly as the sensitive attribute.
    df["gender_like"] = df["A9"].astype(int)
    return df.drop(columns=["A15"])


def convert_compas_two_years() -> pd.DataFrame:
    df = pd.read_csv(RAW / "compas.csv")
    df = df[COMPAS_KEEP].dropna(subset=["race", "two_year_recid"])
    df["race"] = df["race"].replace({"African-American": "black", "Caucasian": "white"})
    return df


def main():
    print("downloading from UCI ML Repository + ProPublica ...")
    for name, url in SOURCES.items():
        fetch(name, url)

    for convert in (convert_adult, convert_german, convert_taiwan_credit,
                    convert_compas_two_years, convert_bank_marketing,
                    convert_student_performance, convert_heart,
                    convert_communities, convert_australian):
        df = convert()
        out = config.DATA_DIR / f"{convert.__name__.removeprefix('convert_')}.csv"
        df.to_csv(out, index=False)

    print("\nconverted -> validated:")
    for out, label, sensitive in SPECS:
        df = data.load_csv(config.DATA_DIR / out)
        problems = data.validate_dataset(df, [sensitive], label)
        status = "OK " if not problems else "!! " + "; ".join(problems)
        rates = df.groupby(sensitive)[label].mean().round(3).to_dict()
        print(f"  {out:20s} {len(df):>6} rows  {status}  {label} rate by {sensitive}: {rates}")

    print("\ntrain a model on any of them:")
    print("  python scripts/train_demo_model.py --data data/adult.csv --label income_gt_50k --sensitive sex")
    print("assess it:")
    print("  python scripts/run_assessment.py --data data/adult.csv --label income_gt_50k "
          "--sensitive sex --model models/adult_logistic_regression.pkl")


if __name__ == "__main__":
    main()
