import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

os.chdir('D:\Proyectos_py/notebooks/Kaggle Courses/Intermediate Machine Learning')
print(os.getcwd())

# Leer datos
data = pd.read_csv("melb_data.csv")

# Selecciona subconjunto de predictores
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Selecciona objetivo
y = data.Price

# Separa los datos en conjutos de entrenamientos y validación
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

print(X_train.head())

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))