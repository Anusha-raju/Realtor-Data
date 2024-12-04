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

#%%[markdown]
# Plotting the distribution of beds variable
#%%

plt.hist(realtordata_clean['bed'], bins=30, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of Bed count')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

#%% [markdown]
# There are a few outliers in the bed count variable, hence removing outliers w.r.t bed count (only the upper bound).

#%%
upper_bound = 200
realtordata_no_outliers = realtordata_clean[(realtordata_clean['bed'] <= upper_bound)]

#%% [markdown]
# imputing a random value between the max and min of bed count.

#%%
random_values = np.random.uniform(realtordata_no_outliers['bed'].min(), realtordata_no_outliers['bed'].max(), size=realtordata_no_outliers['bed'].isna().sum())
realtordata_no_outliers['bed'].loc[realtordata_no_outliers['bed'].isna()] = random_values
print("The mean of bed variable before imputing:  ",realtordata_clean.describe()['bed']['mean'] )
print("The mean of bed variable after imputing:  ", realtordata_no_outliers.describe()['bed']['mean'])

#%%
# The mean of bed variable remains nearly same.

#%%[markdown]
# Plotting the distribution of bath variable
#%%

plt.hist(realtordata_clean['bath'], bins=40, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of Bath count')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

#%% [markdown]
# There are a few outliers in the bath count variable, hence removing outliers w.r.t bath count (only the upper bound).

#%%
upper_bound = 300
realtordata_no_outliers = realtordata_no_outliers[(realtordata_no_outliers['bath'] <= upper_bound)]

#%% [markdown]
# Imputing a random value between the max and min of bath count.

#%%
random_values = np.random.uniform(realtordata_no_outliers['bath'].min(), realtordata_no_outliers['bath'].max(), size=realtordata_no_outliers['bath'].isna().sum())
realtordata_no_outliers['bath'].loc[realtordata_no_outliers['bath'].isna()] = random_values
print("The mean of bath variable before imputing:  ",realtordata_clean.describe()['bath']['mean'] )
print("The mean of bath variable after imputing:  ", realtordata_no_outliers.describe()['bath']['mean'])
#%%
# The mean of bath variable remains nearly same.

#%%


#%%[markdown]
# Plotting the distribution of acre_lot variable
#%%

plt.hist(realtordata_no_outliers['acre_lot'], bins=40, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of acre_lot')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

#%% [markdown]
# Imputing the mean value for acre_lot
#%%
mean_value = realtordata_no_outliers['acre_lot'].mean()
realtordata_no_outliers['acre_lot'].fillna(mean_value, inplace=True)
print("The mean of acre_lot variable before imputing:  ",realtordata_clean.describe()['acre_lot']['mean'] )
print("The mean of acre_lot variable after imputing:  ", realtordata_no_outliers.describe()['acre_lot']['mean'])
#%%
# The mean of acre_lot variable remains nearly same.
