import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
RandomForestRegModel = RandomForestRegressor()
RandomForestRegModel.fit(x,y)
X_marks=[[70]]
print(RandomForestRegModel.predict(X_marks))
