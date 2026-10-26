import csv
from urllib.request import urlopen
from io import StringIO
import random
from src.NNA.engine.BaseArena import BaseArena

#This version includes more features than our earlier versions were ignoring.


class Real_Titanic_7Features(BaseArena):
    """
    Real data. Titanic survival from seven features: the four raw ones plus
    SibSp, Parch and Embarked as an ordinal.
    Second rung: does more raw signal help before any encoding work?
    """
    def __init__(self, num_samples=None):
        self.num_samples = num_samples

    def generate_training_data(self):
        data = Real_Titanic_7Features.load_titanic_data_expanded()

        if self.num_samples:
            if self.num_samples < len(data):
                data = random.sample(data, self.num_samples)

        labels = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Outcome"]
        return data, labels, ["Died", "Survived"]

    @staticmethod
    def load_titanic_data_expanded():
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

                    # Encode embarkation port: S=0, C=1, Q=2
                    embarked_str = row["embarked"].strip().upper()
                    if embarked_str == "S":
                        embarked = 0
                    elif embarked_str == "C":
                        embarked = 1
                    elif embarked_str == "Q":
                        embarked = 2
                    else:
                        continue

                    survived = int(row["survived"].strip())

                    data.append((pclass, sex, age, sibsp, parch, fare, embarked, survived))
            except (ValueError, KeyError, AttributeError):
                continue

        return data
