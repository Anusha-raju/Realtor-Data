#%%


#############
## Imports ##
#############


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#



# %%


###############
## Load Data ##
###############


realtordata = pd.read_csv("c:\\Users\\Cashapona\\Documents\\GWU\\Data Mining\\Final Project\\relator-data.csv")
realtordata.head()


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#



# %%


################
## Clean Data ##
################


# Peak into the structure of the dataset

realtordata.info()


# Dropping the prev_sold_date column as it does not contribute to the analysis.

realtordata.drop(columns = ['prev_sold_date', 'street'], axis=1, inplace=True)


#%%
# Descriptive statistics of numerical columns of Realtor data.

realtordata[['price', 'bed', 'bath', 'acre_lot', 'house_size']].describe()


#%%
# The total number of missing values in the dataset.

print(realtordata.isnull().sum())


#%%
# Dropping data points where brokered_by,price,city,state,zip_code are missing as they constitute a very small number of the dataset.

realtordata_clean = realtordata.dropna(subset=['brokered_by','price','city','state','zip_code'])


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#



# %%


########################
## Outliers and Nulls ##
########################


# Since the missing values in bed, bath, acre_lot and house_size are significantly high, the values will be imputed.


#%%
# BED

# Visualizing the distribution of bed before outlier removal

plt.hist(realtordata_clean['bed'], bins=30, color='skyblue', edgecolor='black') 
plt.yscale('log')
plt.title('Distribution of Bed Count Before Removing Outliers')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

# There are evidently visible outliers in the bed count variable, hence 
# removing outliers w.r.t bed count (only the upper bound).

# Calculating upper bound of the variable

Q1 = realtordata_clean['bed'].quantile(0.25)
Q3 = realtordata_clean['bed'].quantile(0.75)

IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

no_outliers = realtordata_clean[(realtordata_clean['bed'] <= upper_bound)]

# Imputing a random value between the max and min of bed count variable.

random_values = np.random.uniform(no_outliers['bed'].min(), no_outliers['bed'].max(), size=no_outliers['bed'].isna().sum())
no_outliers.loc[no_outliers['bed'].isna(), 'bed'] = random_values

# Visualizing the distribution of bed after outlier removal

sns.histplot(no_outliers['bed'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Bed Count After Removing Outliers')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.show()


#%% 
# BATH

# Visualizing the distribution of bath count before outlier removal

plt.hist(realtordata_clean['bath'], bins=30, color='skyblue', edgecolor='black')
plt.yscale('log') 
plt.title('Distribution of Bath count')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

# Calculating upper bound of the variable

Q1_bath = realtordata_clean['bath'].quantile(0.25)
Q3_bath = realtordata_clean['bath'].quantile(0.75)
IQR_bath = Q3_bath - Q1_bath
upper_bound_bath = Q3_bath + 1.5 * IQR_bath

# Removing outliers for bath variable

no_outliers = no_outliers[no_outliers['bath'] <= upper_bound_bath]

# Imputing a random value between the max and min of bath variable count.

random_values = np.random.uniform(no_outliers['bath'].min(), no_outliers['bath'].max(), size=no_outliers_bath['bath'].isna().sum())
no_outliers.loc[no_outliers['bath'].isna(),'bath'] = random_values

# Visualizing the distribution of bath after outlier removal

sns.histplot(no_outliers['bath'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Bath Count After Removing Outliers')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Frequency')
plt.show()


# %%
# ACRE_LOT

# Vizualizing distribution of acre_lot variable

plt.hist(no_outliers['acre_lot'], bins=40, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of acre_lot')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

print("The mean of acre_lot variable before imputing:  ", no_outliers.describe()['acre_lot']['mean'] )

# Imputing the mean value for acre_lot

mean_value = no_outliers['acre_lot'].mean()
no_outliers['acre_lot'].fillna(mean_value)

print("The mean of acre_lot variable after imputing:  ", no_outliers.describe()['acre_lot']['mean'])


#%%
# HOUSE_SIZE

# Vizualizing distribution of house_size variable

plt.hist(no_outliers['house_size'], bins=40, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of house_size')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

print("The mean of house_size variable before imputing:  ", no_outliers.describe()['house_size']['mean'] )

# Imputing the mean value for house_size

mean_value = no_outliers['house_size'].mean()
no_outliers['house_size'].fillna(mean_value)

print("The mean of house_size variable after imputing:  ", no_outliers.describe()['house_size']['mean'])


# %%
no_outliers.info()

# %%
no_outliers[['price', 'bed', 'bath', 'acre_lot', 'house_size']].describe()


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#



# %%


########################
##   Vizualizations   ##
########################


# Plotting to see the top states with highest number of house sales

for_sale_data = no_outliers['status']
state_counts = no_outliers['state'].value_counts() # grouping by

top_5_states = state_counts.head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_states.index, y=top_5_states.values, palette='viridis')

plt.title('Top 5 States with the Highest Number of Houses For Sales', fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Houses For_Sale', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()

# %%

# Plotting to see the top cities with highest number of house sales

for_sale_data = no_outliers['status']
city_counts = no_outliers['city'].value_counts() # grouping by

top_5_cities = city_counts.head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_cities.index, y=top_5_cities.values, palette='Spectral')

plt.title('Top 5 Cities with the Highest Number of Houses For Sales', fontsize=14)
plt.xlabel('City', fontsize=12)
plt.ylabel('Number of Houses For_Sale', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()

# %%

# Correlation Heatmap of variables

correlation_matrix = no_outliers[['price', 'bed', 'bath', 'acre_lot', 'house_size']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#


