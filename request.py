import requests

url = "http://127.0.0.1:3000/predict"
data = [
    {
        "Pclass": 3,
        "Age": 22,
        "SibSp": 2,
        "Parch": 2,
        "Fare": 0,
        "Sex": "male",
        "Embarked": "C",
    }
]
response = requests.post(url, json=data)
print(response.json())
