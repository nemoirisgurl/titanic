# 🚢 Titanic Survival Predictor


## Introduction
This web application uses Gradient Boosting (XGBoost) to predict passenger survival with **80.4 % Accuracy**, deployed via Flask on Render. Trained infomation are from Kaggle competition **[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)**.

## Features
- Titanic Survival Predictor from Gradient Boosting with train.csv and test.csv from **[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)**
- HTML/CSS form to apply infomation to predict
- Plotly survival graph (Coming Soon)

## Prerequisites
1. Python 3.x
2. Pandas, Scikit-Learn, XGBoost
3. Flask
4. Plotly

## Installation
1. Clone a repository
```
git clone
cd 
```
2. Install dependecies
```
pip install -r requirements.txt
```

3. Train a model
```
python train.py
```

4. Run the app
```
python app.py
```
## Model Perfomance
- Accuracy: 80.4%
- Kaggle Score: 0.78468

## Project Structure
```
|   .gitignore
|   app.py # Flask app
|   README.md
|   request.py # Test file
|   requirements.txt # List of dependencies
|   test.csv # CSV File to test a model
|   train.csv # CSV File to build a model
|   train.py # Trains a model
|
+---models # Models
|
+---static
|       styles.css # Stylesheet
|
\---templates
        graph.html # Shows survival histogram
        index.html # Shows form and results.
```
