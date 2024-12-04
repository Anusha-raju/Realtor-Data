#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional

#%%
realtordata = pd.read_csv("realtor-data.csv")

#%%[markdown]
# Preprocessing
#%%
print("Few records from the dataset")
realtordata.head()
#%%
print("Descriptive statistics of numerical columns of Realtor data.")
realtordata.describe()
#%%
print("The total number of missing values in the dataset.")
print(realtordata.isnull().sum())
#%%[markdown]
# Dropping the prev_sold_date column as it does not contribute to the analysis.
#%%
realtordata.drop('prev_sold_date', axis=1, inplace=True)

#%% [markdown]
# dropping data points where brokered_by,price,city,state,zip_code are missing as they constitute a very small number of the dataset.
#%%
realtordata_clean = realtordata.dropna(subset=['brokered_by','price','city','state','zip_code'])
#%%
print("Since the missing values in bed, bath, acre_lot and house_size are significantly high, the values will be imputed.")

