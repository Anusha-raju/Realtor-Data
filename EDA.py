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
realtordata = pd.read_csv("realtor-data.zip.csv")
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
Q1 = realtordata_no_outliers['bath'].quantile(0.25)
Q3 = realtordata_no_outliers['bath'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

realtordata_no_outliers = realtordata_no_outliers[(realtordata_no_outliers['bath'] <= upper_bound)]
#%% [markdown]
# Imputing a random value between the max and min of bath variable count.

random_values = np.random.uniform(realtordata_no_outliers['bath'].min(), realtordata_no_outliers['bath'].max(), size=realtordata_no_outliers['bath'].isna().sum())
realtordata_no_outliers['bath'].loc[realtordata_no_outliers['bath'].isna()] = random_values
#%%
print("The mean of bath variable before imputing:  ",realtordata_clean.describe()['bath']['mean'] )
print("The mean of bath variable after imputing:  ", realtordata_no_outliers.describe()['bath']['mean'])

#%%[markdown]
# Plotting the distribution of bath variable after removing the outliers
#%%
plt.hist(realtordata_no_outliers['bath'], bins=40, color='skyblue', edgecolor='black')
plt.title('Distribution of Bath count')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()
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
#%%
print("The mean of acre_lot variable before imputing:  ",realtordata_no_outliers.describe()['acre_lot']['mean'] )

#%% [markdown]
# Imputing the mean value for acre_lot
#%%
mean_value = realtordata_no_outliers['acre_lot'].mean()
realtordata_no_outliers['acre_lot'].fillna(mean_value, inplace=True)
#%%
print("The mean of acre_lot variable after imputing:  ", realtordata_no_outliers.describe()['acre_lot']['mean'])

#%%[markdown]
# Plotting the distribution of house_size variable
#%%
plt.hist(realtordata_no_outliers['house_size'], bins=40, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Distribution of house_size')
plt.xlabel('Value')
plt.ylabel('Log Frequency')
plt.show()
#%%
print("The mean of house_size variable before imputing:  ",realtordata_no_outliers.describe()['house_size']['mean'] )

#%% [markdown]
# Imputing the median value for house_size
#%%
mean_value = realtordata_no_outliers['house_size'].mean()
realtordata_no_outliers['house_size'].fillna(mean_value, inplace=True)

print("The mean of house_size variable after imputing:  ", realtordata_no_outliers.describe()['house_size']['mean'])

#%% [markdown]
# Broker Analysis

## # Grouping by broker to calculate average price per broker

#%%
broker_performance = realtordata_no_outliers.groupby('brokered_by')['price'].agg(['mean', 'median', 'std', 'count'])
print("Broker Performance (Average Price, Median Price, Std Deviation, Count of Listings):")
print(broker_performance)

#%% [markdown]
## Price by house_size (median price per house size range)

#%%
realtordata_no_outliers['house_size_category'] = pd.cut(realtordata_no_outliers['house_size'], bins=[0, realtordata_clean['house_size'].quantile(0.25), realtordata_clean['house_size'].quantile(0.50), realtordata_clean['house_size'].quantile(0.75)], labels=['Small', 'Medium', 'Large'])
house_size_performance = realtordata_no_outliers.groupby(['brokered_by', 'house_size_category'])['price'].agg(['mean', 'median'])
print("\nBroker Performance by House Size:")
print(house_size_performance)




#%%[markdown]
# Price by number of bedrooms
#%%
realtordata_no_outliers['bed_category'] = pd.cut(realtordata_no_outliers['bed'], bins=list(range(int(realtordata_no_outliers['bed'].min()-1), int(realtordata_no_outliers['bed'].max())))
, labels=list(range(int(realtordata_no_outliers['bed'].min()), int(realtordata_no_outliers['bed'].max())))
)
bed_performance = realtordata_no_outliers.groupby(['brokered_by', 'bed_category'])['price'].agg(['mean', 'median'])
print("\nBroker Performance by Number of Bedrooms:")
print(bed_performance)
#%% [markdown] Price by lot size (acre_lot)
#%%

realtordata_no_outliers['lot_size_category'] = pd.cut(realtordata_no_outliers['acre_lot'], bins=[0, realtordata_clean['acre_lot'].quantile(0.25), realtordata_clean['acre_lot'].quantile(0.50), realtordata_clean['acre_lot'].quantile(0.75)], labels=['Small', 'Medium', 'Large'])
lot_size_performance = realtordata_no_outliers.groupby(['brokered_by', 'lot_size_category'])['price'].agg(['mean', 'median'])
print("\nBroker Performance by Lot Size:")
print(lot_size_performance)
#%%[markdown]
# Segment wise analysis
#%%
# brokers with the highest average prices
top_brokers_by_price = broker_performance.sort_values(by='mean', ascending=False).head()

# brokers that perform best in specific categories
top_broker_for_large_houses = house_size_performance.loc[house_size_performance['mean'].idxmax()]
top_broker_for_larger_beds = bed_performance.loc[bed_performance['mean'].idxmax()]
top_broker_for_large_lots = lot_size_performance.loc[lot_size_performance['mean'].idxmax()]

# Output Results
print("\nTop Brokers by Average Price:")
print(top_brokers_by_price)

print("\nTop Broker for Large Houses:")
print(top_broker_for_large_houses)

print("\nTop Broker for Larger Beds:")
print(top_broker_for_larger_beds)

print("\nTop Broker for Large Acre Lots:")
print(top_broker_for_large_lots)

#%%[markdown]
# Modelling- Broker Analysis

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

#%%

# Encode categorical variables
label_encoder_city = LabelEncoder()
label_encoder_state = LabelEncoder()
label_encoder_broker = LabelEncoder()

realtordata_no_outliers['city'] = label_encoder_city.fit_transform(realtordata_no_outliers['city'])
realtordata_no_outliers['state'] = label_encoder_state.fit_transform(realtordata_no_outliers['state'])
realtordata_no_outliers['brokered_by'] = label_encoder_broker.fit_transform(realtordata_no_outliers['brokered_by'])


#%%
# Select features for clustering
features = ['bed', 'bath', 'acre_lot', 'house_size', 'city', 'state', 'brokered_by']
X = realtordata_no_outliers[features]

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%%

# Kmeans clustering
kmeans = KMeans(n_clusters=1000, random_state=1)
realtordata_no_outliers['cluster'] = kmeans.fit_predict(X_scaled)
#%%
import pickle

# Save the model
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
#%%

# finding the broker with highest price in a cluster
cluster_info = realtordata_no_outliers.groupby('cluster').agg({'price': 'max', 'brokered_by': 'first'}).reset_index()
cluster_info.to_csv('cluster_info.csv', index=False)
#%%
# Predicting the highest price and broker for a new house
def predict_price_and_broker(new_house_features):
    # Standardize the new house data
    new_house_scaled = scaler.transform([new_house_features])

    # Predict the cluster for the new house
    predicted_cluster = loaded_kmeans.predict(new_house_scaled)[0]

    # Find the highest price and the broker for the predicted cluster
    cluster_data = cluster_info[cluster_info['cluster'] == predicted_cluster]
    highest_price = cluster_data['price'].values[0]
    best_broker = label_encoder_broker.inverse_transform(cluster_data['brokered_by'].values)[0]

    return highest_price, best_broker



#%%
cluster_info = pd.read_csv("cluster_info.csv")

with open('kmeans.pkl', 'rb') as f:
    loaded_kmeans = pickle.load(f)
# Example: Predict for a new house
new_house_data = [3, 3, 0.12, 1200, label_encoder_city.transform(['Adjuntas'])[0], label_encoder_state.transform(['Puerto Rico'])[0], 1]  # New house features
highest_price, best_broker = predict_price_and_broker(new_house_data)

print(f"Predicted Highest Price: ${highest_price}")
print(f"Best Broker (ID): {best_broker}")




