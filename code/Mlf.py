###mlflow 다뤄보기



import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
# print(db)
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)






# import os
# from random import random, randint
# from mlflow import log_metric, log_param, log_params, log_artifacts

# if __name__ == "__main__":
#     # Log a parameter (key-value pair)
#     log_param("config_value", 20)

#     # Log a dictionary of parameters
#     log_params({"param1": 111, "param2": 222})
#     log_param("monster", 19)

#     # Log a metric; metrics can be updated throughout the run
#     log_metric("accuracy", 10.2)
#     log_metric("accuracy", 12)
#     log_metric("accuracy", 8.7)

