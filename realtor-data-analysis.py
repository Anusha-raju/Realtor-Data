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


realtordata = pd.read_csv("realtor-data.zip.csv")
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

random_values = np.random.uniform(no_outliers['bath'].min(), no_outliers['bath'].max(), size=no_outliers['bath'].isna().sum())
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
no_outliers['acre_lot'].fillna(mean_value, inplace=True)


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
no_outliers['house_size'].fillna(mean_value, inplace=True)

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

data = no_outliers

# Plotting to see the top states with highest number of house sales

for_sale_data = data['status']
state_counts = data['state'].value_counts() # grouping by

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

for_sale_data = data['status']
city_counts = data['city'].value_counts() # grouping by

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

correlation_matrix = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

##################################################
#<<<<<<<<<<<<<<<< BROKER ANALYSIS >>>>>>>>>>>>>>>>#


#%% [markdown]
#### Grouping by broker to calculate average price per broker

#%%

broker_performance = no_outliers.groupby('brokered_by')['price'].agg(['mean', 'median', 'std', 'count'])
print("Broker Performance (Average Price, Median Price, Std Deviation, Count of Listings):")
print(broker_performance)

#%%[markdown]

# This code aggregates real estate data by the broker and computes the average (mean), median, standard deviation (std), and count of listings.
# Understanding broker performance helps identify which brokers are handling the highest value properties and their consistency in price.

#%% [markdown]
#### Price by house_size (median price per house size range)

#%%
no_outliers['house_size_category'] = pd.cut(no_outliers['house_size'], bins=[0, no_outliers['house_size'].quantile(0.25), no_outliers['house_size'].quantile(0.50), no_outliers['house_size'].quantile(0.75)], labels=['Small', 'Medium', 'Large'])
house_size_performance = no_outliers.groupby(['brokered_by', 'house_size_category'])['price'].agg(['mean', 'median'])
print("\nBroker Performance by House Size:")
print(house_size_performance)

#%%[markdown]
# The data is categorized into 'Small', 'Medium', and 'Large' based on house size quantiles. The average price by broker and house size category is then calculated.
# Helps identify if brokers specialize in specific types of houses (small, medium, or large) and how that impacts pricing.

#%% [markdown]
#### Price by Number of Bedrooms (Categorizing Bedrooms)

#%%
no_outliers['bed_category'] = pd.cut(no_outliers['bed'], bins=list(range(int(no_outliers['bed'].min()-1), int(no_outliers['bed'].max())))
, labels=list(range(int(no_outliers['bed'].min()), int(no_outliers['bed'].max())))
)
bed_performance = no_outliers.groupby(['brokered_by', 'bed_category'])['price'].agg(['mean', 'median'])
print("\nBroker Performance by Number of Bedrooms:")
print(bed_performance)

#%% [markdown]
# Bedrooms are categorized into ranges, and performance is assessed by broker for each category.
# This helps to analyze how brokers perform with properties of different bedroom counts, which is useful for targeting specific buyer needs.

#%% [markdown]
#### Price by lot size (acre_lot)

#%%
no_outliers['lot_size_category'] = pd.cut(no_outliers['acre_lot'], bins=[0, no_outliers['acre_lot'].quantile(0.25), no_outliers['acre_lot'].quantile(0.50), no_outliers['acre_lot'].quantile(0.75)], labels=['Small', 'Medium', 'Large'])
lot_size_performance = no_outliers.groupby(['brokered_by', 'lot_size_category'])['price'].agg(['mean', 'median'])
print("\nBroker Performance by Lot Size:")
print(lot_size_performance)
#%% [markdown]
# Similar to the house size, properties are grouped by lot size and broker performance is calculated.
# This gives insight into how brokers handle large versus small lots, which can influence property pricing strategies.
#%% [markdown]
#### Segment-wise Analysis of Brokers
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

#%%
########################
##   MODELLING   ##
########################

########################
##   IMPORTS   ##
########################

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
#%%[markdown]
# In the below colab file, we train a ML model that will help in identifying the most suitable broker for a given property based on historical performance and clustering.
#
#To explore the model training process, you can access the full code and notebook for the model training on Google Colab using the link below:
#
#[**Realtor Data Clusttering**](https://colab.research.google.com/drive/1NedKECbrV9Uat5GMw3Up_8gZfsl5_YCe?usp=sharing)
#
#[https://colab.research.google.com/drive/1NedKECbrV9Uat5GMw3Up_8gZfsl5_YCe?usp=sharing](https://colab.research.google.com/drive/1NedKECbrV9Uat5GMw3Up_8gZfsl5_YCe?usp=sharing)
#%% [markdown]
# Encode categorical variables
#%%
no_outliers['zip_code'] = no_outliers['zip_code'].astype(int)
no_outliers['brokered_by'] = no_outliers['brokered_by'].astype(int)
label_encoder_city = LabelEncoder()
label_encoder_state = LabelEncoder()
label_encoder_broker = LabelEncoder()

no_outliers['city'] = label_encoder_city.fit_transform(no_outliers['city'])
no_outliers['state'] = label_encoder_state.fit_transform(no_outliers['state'])
no_outliers['brokered_by'] = label_encoder_broker.fit_transform(no_outliers['brokered_by'])
#%% [markdown]
# This preprocessing step converts non-numeric data (city, state, broker) into a form suitable for clustering, helping identify broker performance patterns based on multiple features.

#%%
# Select features for clustering
features = ['bed', 'bath', 'acre_lot', 'house_size', 'city', 'state', 'brokered_by']
X = no_outliers[features]

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%% [markdown]
#### Elbow Method for selecting the optimal number of clusters
#%%
wcss = []
for k in range(1, 100, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
#%%[markdown]

# The plot shows the Within-Cluster Sum of Squares (WCSS) for different values of k, ranging from 1 to 100, with steps of 10.
# Choosing number of clusters(n_clusters) is 40, as observed from the plot. 
#%%
# Kmeans clustering
kmeans = KMeans(n_clusters=40, random_state=1)
no_outliers['cluster'] = kmeans.fit_predict(X_scaled)
#%%
import pickle

# Save the model
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# finding the broker with highest price in a cluster
cluster_info = no_outliers.groupby('cluster').agg({'price': 'max', 'brokered_by': 'first'}).reset_index()
cluster_info.to_csv('cluster_info.csv', index=False)
#%% [markdown]
# The data is standardized to ensure that all features contribute equally to clustering. KMeans is used to group similar data points into clusters.
# Clustering helps segment the data into groups of similar properties, identifying patterns in how brokers handle different types of properties.

#%% [markdown]
# Loading the model
#%%
cluster_info = pd.read_csv("cluster_info.csv")


with open('kmeans.pkl', 'rb') as f:
    loaded_kmeans = pickle.load(f)
#%% [markdown]
#### Clustering Evaluation using Silhouette Score & Davies Bouldin score
#%%
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, loaded_kmeans.labels_)
print("For n_clusters =", 40,
          "The average silhouette_score is :", silhouette_avg)



db_score = davies_bouldin_score(X_scaled, loaded_kmeans.labels_)
print(f"Davies-Bouldin Index: {db_score:.4f}")

#%% [markdown]
# Predicting the highest price and broker for a new house
#%%
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
#%% [markdown]
# This function predicts the highest possible price and best broker for a new house by first identifying the cluster it belongs to.
# This predictive model helps realtors or analysts estimate prices and identify brokers who are most likely to secure the highest sale for a new listing based on similar past data.
#%% [markdown]
# Example: Predict for a new house
new_house_data = [3, 3, 0.12, 1200, label_encoder_city.transform(['Adjuntas'])[0], label_encoder_state.transform(['Puerto Rico'])[0], 1]  # New house features
highest_price, best_broker = predict_price_and_broker(new_house_data)

print(f"Predicted Highest Price: ${highest_price}")
print(f"Best Broker (ID): {best_broker}")

#%% [markdown]
# Insights- Broker Analysis
# - This model helps realtors predict the highest sale price for new listings based on historical data, allowing them to set competitive prices and identify brokers with the highest likelihood of closing a deal at that price point.
# - By selecting the "best broker" based on past performance in similar property clusters, it ensures that the listing is assigned to a broker who is likely to perform well, increasing the chances of a successful sale.


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#