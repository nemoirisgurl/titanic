import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

TITANIC_DATA = pd.read_csv("train.csv")
TEST_DATA = pd.read_csv("test.csv")
ESTIMATORS = [i * 10 for i in range(1, 21)]

male_survival_rate = TITANIC_DATA.loc[TITANIC_DATA["Sex"] == "male", "Survived"].mean()
female_survival_rate = TITANIC_DATA.loc[
    TITANIC_DATA["Sex"] == "female", "Survived"
].mean()

first_class_survival_rate = TITANIC_DATA.loc[
    TITANIC_DATA["Pclass"] == 1, "Survived"
].mean()
second_class_survival_rate = TITANIC_DATA.loc[
    TITANIC_DATA["Pclass"] == 2, "Survived"
].mean()
third_class_survival_rate = TITANIC_DATA.loc[
    TITANIC_DATA["Pclass"] == 3, "Survived"
].mean()

print("Male survival rate:", male_survival_rate)
print("Female survival rate:", female_survival_rate)
print("First class survival rate:", first_class_survival_rate)
print("Second class survival rate:", second_class_survival_rate)
print("Third class survival rate:", third_class_survival_rate)

TITANIC_DATA["Age"] = TITANIC_DATA["Age"].fillna(TITANIC_DATA["Age"].median())
TEST_DATA["Age"] = TEST_DATA["Age"].fillna(TITANIC_DATA["Age"].median())
TITANIC_DATA["FamilySize"] = TITANIC_DATA["SibSp"] + TITANIC_DATA["Parch"] + 1
TEST_DATA["FamilySize"] = TEST_DATA["SibSp"] + TEST_DATA["Parch"] + 1
TEST_DATA["Fare"] = TEST_DATA["Fare"].fillna(TEST_DATA["Fare"].median())
TITANIC_DATA["Pclass"] = TITANIC_DATA["Pclass"].astype(str)
TEST_DATA["Pclass"] = TEST_DATA["Pclass"].astype(str)
TITANIC_DATA["Embarked"] = TITANIC_DATA["Embarked"].fillna(
    TITANIC_DATA["Embarked"].mode()[0]
)
TEST_DATA["Embarked"] = TEST_DATA["Embarked"].fillna(TITANIC_DATA["Embarked"].mode()[0])


for df in [TITANIC_DATA, TEST_DATA]:
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

categorical_cols = [
    cname for cname in TITANIC_DATA.columns if TITANIC_DATA[cname].dtype == "object"
]
print("Categorical columns:", categorical_cols)

oh_encoder = OneHotEncoder(drop="first", sparse_output=False)


def train_model(n=10, to_csv=False):
    features = ["Pclass", "Age", "Fare", "Sex", "Embarked", "FamilySize", "Title"]
    x = pd.get_dummies(TITANIC_DATA[features])
    y = TITANIC_DATA["Survived"]
    x_test = pd.get_dummies(TEST_DATA[features])
    x, x_test = x.align(x_test, join="left", axis=1, fill_value=0)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    model = XGBClassifier(
        n_estimators=n, random_state=1, learning_rate=0.05, max_depth=3
    )
    model.fit(x_train, y_train)
    score = model.score(x_valid, y_valid)
    mae = 1 - score
    if to_csv:
        model.fit(x, y)
        predictions = model.predict(x_test)
        output = pd.DataFrame(
            {"PassengerId": TEST_DATA["PassengerId"], "Survived": predictions}
        )
        model_survival_rate = output["Survived"].mean()
        print("Model Survival Rate:", model_survival_rate)
        print("Score: ", model.score(x, y))
        output.to_csv("submission.csv", index=False)
        joblib.dump(model, "models/titanicModel.joblib")
        model_columns = list(x_train.columns)
        joblib.dump(model_columns, "models/model_columns.pkl")
    return score, mae


if __name__ == "__main__":
    results = {}
    for estimator in ESTIMATORS:
        score, mae = train_model(n=estimator)
        results[estimator] = (score, mae)
    s = pd.DataFrame(results, index=["R^2 Score", "Mean Absolute Error"]).T
    print(s)

    best_estimator = s["R^2 Score"].idxmax()
    best_mae = s["Mean Absolute Error"].idxmin()
    print(f"Best Estimator: {best_estimator}")
    print(f"Best Mean Absolute Error: {best_mae}")

    train_model(n=best_estimator, to_csv=True)
