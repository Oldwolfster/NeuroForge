import csv
from urllib.request import urlopen
from io import StringIO
import random

from src.NNA.engine.BaseArena import BaseArena


def load_titanic_data_onehot():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    response = urlopen(url)
    csv_text = response.read().decode('utf-8')
    reader = csv.DictReader(StringIO(csv_text))

    data = []
    for row in reader:
        try:
            # Ensure no missing values in selected fields
            required_fields = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
            if all(row.get(k) is not None and row[k].strip() != "" for k in required_fields):
                # Convert and encode values
                pclass = int(row["pclass"].strip())

                sex_str = row["sex"].strip().lower()
                sex = 0 if sex_str == "male" else 1 if sex_str == "female" else None
                if sex is None:
                    continue

                age = float(row["age"].strip())
                sibsp = int(row["sibsp"].strip())
                parch = int(row["parch"].strip())
                fare = float(row["fare"].strip())

                # One-hot encode embarkation port
                embarked_str = row["embarked"].strip().upper()
                embarked_S = 1 if embarked_str == "S" else 0
                embarked_C = 1 if embarked_str == "C" else 0
                embarked_Q = 1 if embarked_str == "Q" else 0

                # Skip if none of the valid ports
                if embarked_S + embarked_C + embarked_Q == 0:
                    continue

                survived = int(row["survived"].strip())

                data.append((pclass, sex, age, sibsp, parch, fare,
                             embarked_S, embarked_C, embarked_Q, survived))
        except (ValueError, KeyError, AttributeError):
            continue

    return data


class Arena_TitanicSurvivors_Real(BaseArena):
    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        data = load_titanic_data_onehot()

        if self.max_rows:
            if self.max_rows < len(data):
                data = random.sample(data, self.max_rows)

        labels = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                  "Embarked_S", "Embarked_C", "Embarked_Q", "Outcome"]
        return data, labels, ["Died", "Survived"]