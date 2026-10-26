import csv
from urllib.request import urlopen
from io import StringIO
import random
import re
from src.NNA.engine.BaseArena import BaseArena

# Note: Assuming these imports exist in your engine based on your snippet
# If Scaler_Standard or Scaler_LogStandard don't exist, fall back to MinMax variants.



def load_titanic_data_v8():
    """
    Titanic V8 (Lean Strategy):
    Removes redundant features (Deck, AgeBin, Pclass_Sex) to reduce noise.
    Prioritizes 'Title' over 'Sex' to capture social status + gender.
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    response = urlopen(url)
    csv_text = response.read().decode('utf-8')
    reader = csv.DictReader(StringIO(csv_text))

    raw_data = list(reader)

    # ========================================================================
    # 1. IMPUTATION CALCULATIONS
    # ========================================================================

    # Helpers for Title extraction
    def get_title(name):
        match = re.search(r' ([A-Za-z]+)\.', name)
        t = match.group(1) if match else "Mr"
        # Standardize
        if t in ["Mlle", "Ms"]: return "Miss"
        if t == "Mme": return "Mrs"
        if t in ["Mr", "Mrs", "Miss", "Master"]: return t
        return "Rare"

    # Aggregators
    age_by_group = {}  # Group by (Title, Pclass)
    fare_by_pclass = {1: [], 2: [], 3: []}
    embarked_counts = {"S": 0, "C": 0, "Q": 0}
    ticket_counts = {}

    # First Pass: Collect Stats
    for row in raw_data:
        # Ticket Freq
        t = row.get("Ticket", "").strip()
        if t: ticket_counts[t] = ticket_counts.get(t, 0) + 1

        # Embarked Mode
        e = row.get("Embarked", "").strip().upper()
        if e in embarked_counts: embarked_counts[e] += 1

        # Fare Stats
        try:
            p = int(row["Pclass"])
            f = float(row["Fare"])
            fare_by_pclass[p].append(f)
        except:
            pass

        # Age Stats
        try:
            p = int(row["Pclass"])
            if row["Age"].strip():
                a = float(row["Age"])
                title = get_title(row["Name"])
                key = (title, p)
                if key not in age_by_group: age_by_group[key] = []
                age_by_group[key].append(a)
        except:
            pass

    # Compute Medians/Modes
    age_medians = {k: sorted(v)[len(v) // 2] for k, v in age_by_group.items()}
    overall_age = 28.0
    fare_medians = {k: sorted(v)[len(v) // 2] for k, v in fare_by_pclass.items()}
    embarked_mode = max(embarked_counts, key=embarked_counts.get)

    # ========================================================================
    # 2. FEATURE CONSTRUCTION
    # ========================================================================
    data = []

    for row in raw_data:
        try:
            # --- TARGET ---
            if not row["Survived"]: continue
            survived = int(row["Survived"])

            # --- CORE INPUTS ---
            pclass = int(row["Pclass"])

            # Title Processing
            title = get_title(row["Name"])

            # Age Imputation
            if row["Age"].strip():
                age = float(row["Age"])
            else:
                age = age_medians.get((title, pclass), overall_age)

            # Fare Processing
            if row["Fare"].strip():
                fare = float(row["Fare"])
            else:
                fare = fare_medians.get(pclass, 14.0)

            # Embarked Imputation
            embarked = row["Embarked"].strip().upper()
            if embarked not in ["S", "C", "Q"]:
                embarked = embarked_mode

            # Derived Features
            sibsp = int(row["SibSp"]) if row["SibSp"] else 0
            parch = int(row["Parch"]) if row["Parch"] else 0
            family_size = sibsp + parch + 1

            ticket = row.get("Ticket", "").strip()
            ticket_freq = ticket_counts.get(ticket, 1)

            # Fare Per Person (Critical for grouping)
            group_size = max(family_size, ticket_freq)
            fare_pp = fare / group_size if group_size > 0 else fare

            # Cabin (Binary is sufficient)
            has_cabin = 1 if row.get("Cabin", "").strip() else 0

            # --- ENCODING ---

            # Title Effect Code (Mr=0, Mrs=1, Miss=2, Master=3, Rare=4)
            # We used 'Rare' as reference (-1,-1,-1,-1) in V7, let's keep that consistency
            t_map = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}
            t_code = t_map.get(title, 4)
            t_feats = effect_code(t_code, 5)  # Returns 4 values

            # Embarked Effect Code (S=0, C=1, Q=2)
            e_map = {"S": 0, "C": 1, "Q": 2}
            e_code = e_map.get(embarked, 0)
            e_feats = effect_code(e_code, 3)  # Returns 2 values

            # --- ASSEMBLY (12 Features) ---
            features = (
                pclass,  # 1
                age,  # 2
                fare_pp,  # 3
                family_size,  # 4
                ticket_freq,  # 5
                has_cabin,  # 6
                t_feats[0], t_feats[1], t_feats[2], t_feats[3],  # 7-10 (Title)
                e_feats[0], e_feats[1],  # 11-12 (Embarked)
                survived  # Target
            )
            data.append(features)

        except Exception:
            continue

    return data


def effect_code(category, num_categories):
    """Returns tuple of length (num_categories - 1). Last cat is all -1."""
    num_features = num_categories - 1
    if category == num_categories - 1:
        return tuple([-1] * num_features)
    else:
        result = [0] * num_features
        if category < num_features:
            result[category] = 1
        return tuple(result)


class Arena_TitanicSurvivors_V8(BaseArena):
    """
    Titanic V8: High-Signal, Low-Noise.
    Inputs: 12
    """

    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        data = load_titanic_data_v8()
        if self.max_rows and self.max_rows < len(data):
            data = random.sample(data, self.max_rows)

        labels = [
            "Pclass", "Age", "FarePP", "FamilySize", "TicketFreq", "HasCabin",
            "Title_Mr", "Title_Mrs", "Title_Miss", "Title_Master",
            "Embarked_S", "Embarked_C",
            "Outcome"
        ]
        return data, labels, ["Died", "Survived"]