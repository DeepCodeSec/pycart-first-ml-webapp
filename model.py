import pandas as pd
import pickle
from pycaret.regression import *
import numpy as np

df = pd.read_csv('insurance.csv')

r2 = setup(df, target='charges',session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True,
           feature_interaction=True, 
           bin_numeric_features= ['age', 'bmi'])

# Model Training and Validation 
lr = create_model('lr')

# plot residuals of trained model**
#plot_model(lr, plot = 'residuals')

# save transformation pipeline and model 
save_model(lr, model_name = 'deployment_28042020')

