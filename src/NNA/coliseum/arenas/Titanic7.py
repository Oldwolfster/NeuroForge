import csv
from urllib.request import urlopen
from io import StringIO
import random
import re
from src.NNA.engine.BaseArena import BaseArena


def load_titanic_data_v7():
    """
    Titanic V7: Full feature engineering based on top Kaggle solutions.
    Uses original Kaggle dataset with Name, Ticket, Cabin columns.
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    response = urlopen(url)
    csv_text = response.read().decode('utf-8')
    reader = csv.DictReader(StringIO(csv_text))

    # ========================================================================
    # FIRST PASS: Collect all data and compute statistics for imputation
    # ========================================================================
    raw_data = []

    # For age imputation by Title + Pclass
    age_by_title_pclass = {}

    # For fare imputation by Pclass
    fare_by_pclass = {1: [], 2: [], 3: []}

    # For embarked mode
    embarked_counts = {"S": 0, "C": 0, "Q": 0}

    # For ticket frequency
    ticket_counts = {}

    for row in reader:
        raw_data.append(row)

        # Extract title for age grouping
        name = row.get("Name", "")
        title = extract_title(name)
        title = standardize_title(title)

        # Collect age by title + pclass
        try:
            pclass = int(row["Pclass"].strip())
            age_str = row["Age"].strip()
            if age_str:
                age = float(age_str)
                key = (title, pclass)
                if key not in age_by_title_pclass:
                    age_by_title_pclass[key] = []
                age_by_title_pclass[key].append(age)
        except (ValueError, KeyError):
            pass

        # Collect fare by pclass
        try:
            if row["Fare"].strip():
                pclass = int(row["Pclass"].strip())
                fare = float(row["Fare"].strip())
                if pclass in fare_by_pclass:
                    fare_by_pclass[pclass].append(fare)
        except (ValueError, KeyError):
            pass

        # Count embarked ports
        try:
            embarked = row["Embarked"].strip().upper()
            if embarked in embarked_counts:
                embarked_counts[embarked] += 1
        except (ValueError, KeyError):
            pass

        # Count ticket occurrences
        ticket = row.get("Ticket", "").strip()
        if ticket:
            ticket_counts[ticket] = ticket_counts.get(ticket, 0) + 1

    # ========================================================================
    # COMPUTE MEDIANS AND MODES
    # ========================================================================

    # Age medians by title + pclass
    age_medians = {}
    for key, ages in age_by_title_pclass.items():
        if ages:
            age_medians[key] = sorted(ages)[len(ages) // 2]

    # Fallback: overall median by pclass only
    age_by_pclass = {1: [], 2: [], 3: []}
    for (title, pclass), ages in age_by_title_pclass.items():
        if pclass in age_by_pclass:
            age_by_pclass[pclass].extend(ages)
    age_medians_pclass = {}
    for pclass, ages in age_by_pclass.items():
        if ages:
            age_medians_pclass[pclass] = sorted(ages)[len(ages) // 2]

    # Overall fallback
    all_ages = []
    for ages in age_by_title_pclass.values():
        all_ages.extend(ages)
    overall_age_median = sorted(all_ages)[len(all_ages) // 2] if all_ages else 28

    # Fare medians by pclass
    fare_medians = {}
    for pclass, fares in fare_by_pclass.items():
        if fares:
            fare_medians[pclass] = sorted(fares)[len(fares) // 2]

    # Embarked mode
    embarked_mode = max(embarked_counts, key=embarked_counts.get)

    # ========================================================================
    # SECOND PASS: Process data with full feature engineering
    # ========================================================================
    data = []

    for row in raw_data:
        try:
            # ------------------------------------------------------------------
            # BASIC FIELDS
            # ------------------------------------------------------------------
            pclass = int(row["Pclass"].strip())

            sex_str = row["Sex"].strip().lower()
            sex = 0 if sex_str == "male" else 1 if sex_str == "female" else None
            if sex is None:
                continue

            sibsp = int(row["SibSp"].strip()) if row["SibSp"].strip() else 0
            parch = int(row["Parch"].strip()) if row["Parch"].strip() else 0

            # Survived (required)
            survived_str = row["Survived"].strip()
            if not survived_str:
                continue
            survived = int(survived_str)

            # ------------------------------------------------------------------
            # TITLE EXTRACTION
            # ------------------------------------------------------------------
            name = row.get("Name", "")
            title = extract_title(name)
            title = standardize_title(title)

            # ------------------------------------------------------------------
            # AGE (impute by Title + Pclass)
            # ------------------------------------------------------------------
            age_str = row["Age"].strip()
            if age_str:
                age = float(age_str)
            else:
                key = (title, pclass)
                age = age_medians.get(key)
                if age is None:
                    age = age_medians_pclass.get(pclass, overall_age_median)

            # ------------------------------------------------------------------
            # FARE (impute by Pclass)
            # ------------------------------------------------------------------
            fare_str = row["Fare"].strip()
            if fare_str:
                fare = float(fare_str)
            else:
                fare = fare_medians.get(pclass, 30.0)

            # ------------------------------------------------------------------
            # EMBARKED (impute with mode)
            # ------------------------------------------------------------------
            embarked_str = row["Embarked"].strip().upper()
            if embarked_str not in ["S", "C", "Q"]:
                embarked_str = embarked_mode

            # ------------------------------------------------------------------
            # FAMILY FEATURES
            # ------------------------------------------------------------------
            family_size = sibsp + parch + 1
            is_alone = 1 if family_size == 1 else 0

            # ------------------------------------------------------------------
            # TICKET FREQUENCY
            # ------------------------------------------------------------------
            ticket = row.get("Ticket", "").strip()
            ticket_freq = ticket_counts.get(ticket, 1)

            # ------------------------------------------------------------------
            # FARE PER PERSON
            # ------------------------------------------------------------------
            group_size = max(family_size, ticket_freq)
            fare_pp = fare / group_size if group_size > 0 else fare

            # ------------------------------------------------------------------
            # CABIN FEATURES
            # ------------------------------------------------------------------
            cabin = row.get("Cabin", "").strip()
            has_cabin = 1 if cabin else 0
            deck = cabin[0].upper() if cabin else "U"
            # Normalize rare decks
            if deck not in ["A", "B", "C", "D", "E", "F", "G", "U"]:
                deck = "U"

            # ------------------------------------------------------------------
            # WOMAN OR CHILD (captures "women and children first")
            # ------------------------------------------------------------------
            is_child = 1 if age <= 16 else 0
            woman_or_child = 1 if (sex == 1 or is_child == 1) else 0

            # ------------------------------------------------------------------
            # AGE BINNING: 0=Child, 1=Young, 2=Adult, 3=Middle, 4=Senior
            # ------------------------------------------------------------------
            if age <= 16:
                age_bin = 0
            elif age <= 32:
                age_bin = 1
            elif age <= 48:
                age_bin = 2
            elif age <= 64:
                age_bin = 3
            else:
                age_bin = 4

            # ------------------------------------------------------------------
            # PCLASS_SEX INTERACTION (6 categories -> 0-5)
            # ------------------------------------------------------------------
            pclass_sex = (pclass - 1) * 2 + sex  # 0-5

            # ------------------------------------------------------------------
            # EFFECT CODING
            # ------------------------------------------------------------------

            # Title: Mr=0, Mrs=1, Miss=2, Master=3, Rare=4
            title_map = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}
            title_code = title_map.get(title, 4)
            title_f1, title_f2, title_f3, title_f4 = effect_code(title_code, 5)

            # Embarked: S=0, C=1, Q=2
            embarked_map = {"S": 0, "C": 1, "Q": 2}
            embarked_code = embarked_map.get(embarked_str, 0)
            embarked_f1, embarked_f2 = effect_code(embarked_code, 3)

            # Deck: A=0, B=1, C=2, D=3, E=4, F=5, G=6, U=7
            deck_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "U": 7}
            deck_code = deck_map.get(deck, 7)
            deck_f1, deck_f2, deck_f3, deck_f4, deck_f5, deck_f6, deck_f7 = effect_code(deck_code, 8)

            # AgeBin: 0-4 (5 categories)
            agebin_f1, agebin_f2, agebin_f3, agebin_f4 = effect_code(age_bin, 5)

            # Pclass_Sex: 0-5 (6 categories)
            pcsex_f1, pcsex_f2, pcsex_f3, pcsex_f4, pcsex_f5 = effect_code(pclass_sex, 6)

            # ------------------------------------------------------------------
            # ASSEMBLE FEATURES
            # ------------------------------------------------------------------
            features = (
                # Core numeric features
                pclass,  # 1
                sex,  # 2
                age,  # 3
                fare_pp,  # 4
                family_size,  # 5
                is_alone,  # 6
                has_cabin,  # 7
                woman_or_child,  # 8
                ticket_freq,  # 9
                # Title effect coding (4 features)
                title_f1, title_f2, title_f3, title_f4,  # 10-13
                # Embarked effect coding (2 features)
                embarked_f1, embarked_f2,  # 14-15
                # Deck effect coding (7 features)
                deck_f1, deck_f2, deck_f3, deck_f4, deck_f5, deck_f6, deck_f7,  # 16-22
                # AgeBin effect coding (4 features)
                agebin_f1, agebin_f2, agebin_f3, agebin_f4,  # 23-26
                # Pclass_Sex effect coding (5 features)
                pcsex_f1, pcsex_f2, pcsex_f3, pcsex_f4, pcsex_f5,  # 27-31
                # Target
                survived  # 32
            )

            data.append(features)

        except (ValueError, KeyError, AttributeError) as e:
            continue

    return data


def extract_title(name):
    """Extract title from name using regex."""
    match = re.search(r' ([A-Za-z]+)\.', name)
    if match:
        return match.group(1)
    return "Mr"  # Default fallback


def standardize_title(title):
    """Standardize titles into 5 categories: Mr, Mrs, Miss, Master, Rare."""
    # Map French titles
    if title in ["Mlle", "Ms"]:
        return "Miss"
    if title == "Mme":
        return "Mrs"

    # Keep common titles
    if title in ["Mr", "Mrs", "Miss", "Master"]:
        return title

    # Everything else is Rare
    return "Rare"


def effect_code(category, num_categories):
    """
    Generate effect coding for a categorical variable.
    Returns (num_categories - 1) values.
    The last category gets all -1s.
    """
    num_features = num_categories - 1
    if category == num_categories - 1:
        # Reference category: all -1s
        return tuple([-1] * num_features)
    else:
        # One-hot style but with -1 for reference
        result = [0] * num_features
        if category < num_features:
            result[category] = 1
        return tuple(result)


class Arena_TitanicSurvivors_V7(BaseArena):
    """
    Titanic V7: Full feature engineering for maximum accuracy.

    Features (31 inputs):
    - Core: Pclass, Sex, Age, FarePP, FamilySize, IsAlone, HasCabin, WomanOrChild, TicketFreq
    - Title (effect coded): 4 features
    - Embarked (effect coded): 2 features
    - Deck (effect coded): 7 features
    - AgeBin (effect coded): 4 features
    - Pclass_Sex (effect coded): 5 features
    """

    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        data = load_titanic_data_v7()

        if self.max_rows:
            if self.max_rows < len(data):
                data = random.sample(data, self.max_rows)

        labels = [
            # Core features
            "Pclass", "Sex", "Age", "FarePP", "FamilySize",
            "IsAlone", "HasCabin", "WomanOrChild", "TicketFreq",
            # Title effect coding
            "Title_F1", "Title_F2", "Title_F3", "Title_F4",
            # Embarked effect coding
            "Embarked_F1", "Embarked_F2",
            # Deck effect coding
            "Deck_F1", "Deck_F2", "Deck_F3", "Deck_F4", "Deck_F5", "Deck_F6", "Deck_F7",
            # AgeBin effect coding
            "AgeBin_F1", "AgeBin_F2", "AgeBin_F3", "AgeBin_F4",
            # Pclass_Sex effect coding
            "PclassSex_F1", "PclassSex_F2", "PclassSex_F3", "PclassSex_F4", "PclassSex_F5",
            # Target
            "Outcome"
        ]

        return data, labels, ["Died", "Survived"]