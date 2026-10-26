import csv
from urllib.request import urlopen
from io import StringIO

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
            if all(row[k] for k in ["pclass", "sex", "age", "fare", "survived"]):
                # Convert and encode values
                pclass = int(row["pclass"])
                sex = 0 if row["sex"] == "male" else 1
                age = float(row["age"])
                fare = float(row["fare"])
                survived = int(row["survived"])

                data.append((pclass, sex, age, fare, survived))
        except ValueError:
            continue  # Skip malformed rows

    return data


class Arena_TitanicSurvivors_Real(BaseArena):
    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        data = load_titanic_data()

        if self.max_rows:
            data = data[:self.max_rows]

        labels = ["Pclass", "Sex", "Age", "Fare", "Outcome"]
        return data, labels, ["Died", "Survived"]
