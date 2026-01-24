import csv
from urllib.request import urlopen
from io import StringIO
import random

from src.NNA.engine.BaseArena import BaseArena


def load_titanic_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    response = urlopen(url)
    csv_text = response.read().decode('utf-8')
    reader = csv.DictReader(StringIO(csv_text))

    data = []
    for row in reader:
        try:
            # Ensure no missing values in selected fields
            if all(row.get(k) is not None and row[k].strip() != "" for k in ["pclass", "sex", "age", "fare", "survived"]):
                # Convert and encode values
                pclass = int(row["pclass"].strip())
                sex_str = row["sex"].strip().lower()
                sex = 0 if sex_str == "male" else 1 if sex_str == "female" else None
                if sex is None:
                    continue
                age = float(row["age"].strip())
                fare = float(row["fare"].strip())
                survived = int(row["survived"].strip())

                data.append((pclass, sex, age, fare, survived))
        except (ValueError, KeyError, AttributeError):
            continue  # Skip malformed rows

    return data


class Arena_TitanicSurvivors_Real(BaseArena):
    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        data = load_titanic_data()

        if self.max_rows:
            if self.max_rows < len(data):
                data = random.sample(data, self.max_rows)
            # else: keep all rows if requested more than available

        labels = ["Pclass", "Sex", "Age", "Fare", "Outcome"]
        return data, labels, ["Died", "Survived"]
