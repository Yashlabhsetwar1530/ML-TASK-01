import joblib

model = joblib.load("salary_model.pkl")
print(model.predict([[6.4]]))
