import joblib
import numpy as np
import pandas as pd
model = joblib.load("model/model.pkl")
def predict(features):
    features = np.array(features, dtype=object).reshape(1, -1)
    columns = pd.read_csv('data/online vs store shopping dataset.csv').columns.tolist()[:-1]
    df = pd.DataFrame(features, columns=columns)
    prediction = model.predict(df)
    return prediction[0]
