import csv
from urllib.request import urlopen
from io import StringIO
import random

from src.NNA.engine.BaseArena import BaseArena




class Real_Titanic_4Features(BaseArena):
    """
    Real data. Titanic survival from four raw features: Pclass, Sex, Age, Fare.
    Rows with any missing value are dropped. First rung of the Titanic ladder --
    the baseline every engineered version is measured against.
    """
    def __init__(self, num_samples=None):
        self.num_samples = num_samples

    def generate_training_data(self):
        data = Real_Titanic_4Features.load_titanic_data()

        if self.num_samples:
            if self.num_samples < len(data):
                data = random.sample(data, self.num_samples)
            # else: keep all rows if requested more than available

        labels = ["Pclass", "Sex", "Age", "Fare", "Outcome"]
        return data, labels, ["Died", "Survived"]

    @staticmethod
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
