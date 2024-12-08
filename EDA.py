#%% [markdown]
# Importing necessary libraries
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


#%%[markdown]
# Loading the dataset 
#%%
plt.figure(figsize=(8, 6))
realtordata = pd.read_csv("/home/anusha/Downloads/realtor-data.zip.csv")
realtordata.head()

#%%[markdown]
# Preprocessing the data

#%%[markdown]
# Descriptive statistics of numerical columns of Realtor data.
#%%
realtordata[['price', 'bed', 'bath', 'acre_lot', 'house_size']].describe()

#%%[markdown]
# The total number of missing values in the dataset.
#%%
print(realtordata.isnull().sum())

#%%[markdown]
# Dropping the prev_sold_date column as it does not contribute to the analysis.
#%%
realtordata.drop(columns = ['prev_sold_date', 'street'], axis=1, inplace=True)

#%% [markdown]
# Dropping data points where brokered_by,price,city,state,zip_code are missing as they constitute a very small number of the dataset.
#%%
realtordata_clean = realtordata.dropna(subset=['brokered_by','price','city','state','zip_code'])

#%%[markdown]
# Since the missing values in bed, bath, acre_lot and house_size are significantly high, the values will be imputed.

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
# There are evidently visible outliers in the bed count variable, hence removing outliers w.r.t bed count (only the upper bound).

#%%

Q1 = realtordata_clean['bed'].quantile(0.25)
Q3 = realtordata_clean['bed'].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
realtordata_no_outliers = realtordata_clean[(realtordata_clean['bed'] <= upper_bound)]

#%% [markdown]
# Imputing a random value between the max and min of bed count variable.
#%%
random_values = np.random.uniform(realtordata_no_outliers['bed'].min(), realtordata_no_outliers['bed'].max(), size=realtordata_no_outliers['bed'].isna().sum())
realtordata_no_outliers['bed'].loc[realtordata_no_outliers['bed'].isna()] = random_values
#%%
print("The mean of bed variable before imputing:  ",realtordata_clean.describe()['bed']['mean'] )
print("The mean of bed variable after imputing:  ", realtordata_no_outliers.describe()['bed']['mean'])

#%% [markdown]
# Plotting the distribution of beds variable after removing the outliers.
#%%
plt.hist(realtordata_no_outliers['bed'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Bed count')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

#%% [markdown]
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
# Imputing a random value between the max and min of bath variable count.

#%%
random_values = np.random.uniform(realtordata_no_outliers['bath'].min(), realtordata_no_outliers['bath'].max(), size=realtordata_no_outliers['bath'].isna().sum())
realtordata_no_outliers['bath'].loc[realtordata_no_outliers['bath'].isna()] = random_values
#%%
print("The mean of bath variable before imputing:  ",realtordata_clean.describe()['bath']['mean'] )
print("The mean of bath variable after imputing:  ", realtordata_no_outliers.describe()['bath']['mean'])
#%%[markdown]
# The mean of bath variable remains nearly same.

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
# Imputing the median value for acre_lot
#%%
median_value = realtordata_no_outliers['acre_lot'].median()
realtordata_no_outliers['acre_lot'].fillna(median_value, inplace=True)
#%%
print("The median of acre_lot variable before imputing:  ",realtordata_clean.describe()['acre_lot']['mean'] )
print("The median of acre_lot variable after imputing:  ", realtordata_no_outliers.describe()['acre_lot']['mean'])

#%%[markdown]
# Plotting the distribution of house_size variable
#%%
plt.hist(realtordata_no_outliers['house_size'], bins=40, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of house_size')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()

#%% [markdown]
# Imputing the median value for house_size
#%%
median_value = realtordata_no_outliers['house_size'].median()
realtordata_no_outliers['house_size'].fillna(median_value, inplace=True)
#%%
print("The median of house_size variable before imputing:  ",realtordata_clean.describe()['house_size']['mean'] )
print("The median of house_size variable after imputing:  ", realtordata_no_outliers.describe()['house_size']['mean'])

#%% [markdown]
# Vizualisations

#%% [markdown]
# Pivoting data to see how many houses are for_sale in each state
# %%
pivot_table = realtordata_no_outliers.pivot_table(
    index='state', 
    columns='status', 
    values='price',  
    aggfunc='count',
    fill_value=0  
)

print(pivot_table)

# %%
for_sale_data = realtordata_no_outliers['status']
state_counts = realtordata_no_outliers['state'].value_counts() # grouping by

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
for_sale_data = realtordata_no_outliers['status']
city_counts = realtordata_no_outliers['city'].value_counts() # grouping by

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

#%%
sns.scatterplot(x='bed', y='price', data=realtordata_no_outliers)
plt.title('Price vs Bedrooms')
plt.show()

#%%
sns.scatterplot(x='house_size', y='price', data=realtordata_no_outliers)
plt.title('Price vs House Size')
plt.show()

#%%
sns.scatterplot(x='acre_lot', y='price', data=realtordata_no_outliers)
plt.title('Price vs Acre Lot')
plt.show()

#%%
corr = realtordata_no_outliers[['price', 'bed', 'bath', 'acre_lot', 'house_size']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()



#%%

