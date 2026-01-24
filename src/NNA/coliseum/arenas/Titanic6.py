import csv
from urllib.request import urlopen
from io import StringIO
import random
from src.NNA.engine.BaseArena import BaseArena


def load_titanic_data_effect_code():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    response = urlopen(url)
    csv_text = response.read().decode('utf-8')
    reader = csv.DictReader(StringIO(csv_text))

    # First pass: collect all data and compute statistics for imputation
    raw_data = []
    age_by_group = {}  # Will store ages by (pclass, sex) for median calculation
    embarked_counts = {"S": 0, "C": 0, "Q": 0}
    fare_by_pclass = {1: [], 2: [], 3: []}

    for row in reader:
        raw_data.append(row)

        # Collect age data by group for median calculation
        try:
            pclass = int(row["pclass"].strip())
            sex_str = row["sex"].strip().lower()
            age_str = row["age"].strip()

            if age_str:  # If age exists
                age = float(age_str)
                key = (pclass, sex_str)
                if key not in age_by_group:
                    age_by_group[key] = []
                age_by_group[key].append(age)

        except (ValueError, KeyError):
            pass

        # Count embarked ports for mode
        try:
            embarked = row["embarked"].strip().upper()
            if embarked in embarked_counts:
                embarked_counts[embarked] += 1
        except (ValueError, KeyError):
            pass

        # Collect fare by pclass for median
        try:
            if row["fare"].strip():
                pclass = int(row["pclass"].strip())
                fare = float(row["fare"].strip())
                if pclass in fare_by_pclass:
                    fare_by_pclass[pclass].append(fare)
        except (ValueError, KeyError):
            pass

    # Calculate medians for each group
    age_medians = {}
    for key, ages in age_by_group.items():
        age_medians[key] = sorted(ages)[len(ages) // 2]

    # Calculate fare medians by pclass
    fare_medians = {}
    for pclass, fares in fare_by_pclass.items():
        if fares:
            fare_medians[pclass] = sorted(fares)[len(fares) // 2]

    # Find mode for embarked
    embarked_mode = max(embarked_counts, key=embarked_counts.get)

    # Overall median age as fallback
    all_ages = []
    for ages in age_by_group.values():
        all_ages.extend(ages)
    overall_age_median = sorted(all_ages)[len(all_ages) // 2] if all_ages else 28

    # Second pass: process data with imputation
    data = []
    for row in raw_data:
        try:
            # Get pclass (required for other imputations)
            pclass = int(row["pclass"].strip())

            # Get/impute sex
            sex_str = row["sex"].strip().lower()
            sex = 0 if sex_str == "male" else 1 if sex_str == "female" else None
            if sex is None:  # Skip if sex is invalid
                continue

            # Get/impute age
            age_str = row["age"].strip()
            if age_str:
                age = float(age_str)
            else:
                # Impute age based on pclass and sex
                key = (pclass, sex_str)
                age = age_medians.get(key, overall_age_median)

            # Get sibsp and parch (usually not missing)
            sibsp = int(row["sibsp"].strip()) if row["sibsp"].strip() else 0
            parch = int(row["parch"].strip()) if row["parch"].strip() else 0

            # Get/impute fare
            fare_str = row["fare"].strip()
            if fare_str:
                fare = float(fare_str)
            else:
                # Impute fare based on pclass
                fare = fare_medians.get(pclass, 30.0)  # Default fallback

            # Get/impute embarked
            embarked_str = row["embarked"].strip().upper()
            if embarked_str not in ["S", "C", "Q"]:
                embarked_str = embarked_mode

            # One-hot encode embarkation port - switching to effect code
            #embarked_S = 1 if embarked_str == "S" else 0
            #embarked_C = 1 if embarked_str == "C" else 0
            #embarked_Q = 1 if embarked_str == "Q" else 0

            if embarked_str == "S":
                embarked_F1= 1
                embarked_F2= 0
            if embarked_str == "C":
                embarked_F1 = 0
                embarked_F2 = 1
            if embarked_str == "Q":
                embarked_F1 = -1
                embarked_F2 = -1

            # Get survived status (required - skip if missing)
            survived_str = row["survived"].strip()
            if not survived_str:
                continue
            survived = int(survived_str)

            data.append((pclass, sex, age, sibsp, parch, fare,
                         #embarked_S, embarked_C, embarked_Q,
                         embarked_F1, embarked_F2,
                         survived))

        except (ValueError, KeyError, AttributeError):
            continue

    return data


class Arena_TitanicSurvivors_Real(BaseArena):
    """
    Version 6 switches from one hot encoding to effect coding
    """

    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        data = load_titanic_data_effect_code()

        if self.max_rows:
            if self.max_rows < len(data):
                data = random.sample(data, self.max_rows)

        labels = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                  "Embarked_F1", "Embarked_F2", "Outcome"]
        return data, labels, ["Died", "Survived"]