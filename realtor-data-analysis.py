#%%


#############
## Imports ##
#############


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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

label_encoder_state = LabelEncoder()
data['state_encoded'] = label_encoder_state.fit_transform(data['state'])

label_encoder_zip = LabelEncoder()
data['zip_code_encoded'] = label_encoder_zip.fit_transform(data['zip_code'])

label_encoder_city = LabelEncoder()
data['city_encoded'] = label_encoder_city.fit_transform(data['city'])

correlation_data = data[['price', 'bed', 'bath', 'acre_lot', 'house_size', 
                         'state_encoded', 'zip_code_encoded', 'city_encoded']]

correlation_matrix = correlation_data.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap Including Categorical Variables')
plt.show()


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#



#%%


###################################
##   Property Price Prediction   ##
###################################


# Linear Regression 1 - Predicting the price of the property only with numerical variables as predictors

# Assigning predictors and target variable

X = data[['bed', 'bath', 'acre_lot', 'house_size']]
y = data['price']  

# Scaling the features to avoid bias

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Training the model

model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set

y_pred_test = model.predict(X_test)

# Making predictions on the training set

y_pred_train = model.predict(X_train)

# Training Set Evaluation

print("Training Set Evaluation:")
print(f"R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")
print("-" * 30)

# Test Set Evaluation

print("Test Set Evaluation:")
print(f"R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")
print("-" * 30)

# Comparing actual vs predicted values

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print(df.head())

# Plotting the values 

plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Insight 

# The low R^2 values for both the training set and the test set indicates that the model is 
# not capturing much of the variability in the property prices suggesting that, numerical 
# predictors alone may not be sufficient to explain the price variations. 
# Additionally, the similar performance on the training and test sets points to underfitting.

#%%

# Linear Regression 2 - Predicting the property price city as a predictor along with numerical variables

# Selecting predictors and target variable

X = data[['bed', 'bath', 'house_size', 'acre_lot', 'city']].copy()  # Explicitly creating a copy to avoid 'SettingWithCopyWarning' because we're modifying a slice of a DataFrame, which can lead to unintended side effects.
y = data['price']

# Encoding the 'city' variable

label_encoder_city = LabelEncoder()
X['city'] = label_encoder_city.fit_transform(X['city']) 

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features

scaler = StandardScaler()
numerical_features = ['bed', 'bath', 'house_size', 'acre_lot']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Training the model

model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set

y_pred_test = model.predict(X_test)

# Making predictions on the training set

y_pred_train = model.predict(X_train)

# Training Set Evaluation

print("Training Set Evaluation:")
print(f"R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")
print("-" * 30)

# Test Set Evaluation

print("Test Set Evaluation:")
print(f"R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")
print("-" * 30)

# Comparing actual vs predicted values

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print(df.head())

# Plotting the values 

plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Insight

# The inclusion of city variable slightly improved the model's predictive ability for the test set, 
# as seen in the increase of R^2 score. This suggests that the city variable captures some location-specific 
# variation in property prices, which was missing in the first model. However, the improvement 
# is relatively very small, with a significant amount of unexplained variability in property prices
# And the R^2 for training set remains low and identical to first model showcasing underfitting.

# Overall insights

# Both the models show very low R^2 values on the training set and marginally better performance on 
# the test sets. This consistent low R^2 indicates that the models fail to capture most of 
# the variability in property prices, suggesting weak or non-existent linear relationships 
# between the predictors and the target variable. While adding city to the model slighty 
# improved test score, it wasn't sufficient to improve model predictions. We can also see the 
# reflection of consistent underfitting.


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

#%%
# Kmeans clustering
kmeans = KMeans(n_clusters=1000, random_state=1)
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
#### Clustering Evaluation using Silhouette Score
#%%
from sklearn.metrics import silhouette_score

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, loaded_kmeans.labels_)
print("For n_clusters =", 1000,
          "The average silhouette_score is :", silhouette_avg)

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

#%%
#############################################
# Import necessary libraries
#############################################
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%
#############################################
# Step 1: Sampling the data for efficiency
#############################################
# Limit data to 1,000 samples per state to ensure a manageable dataset for clustering.
sample_per_state = 1000  # Number of samples per state

sampled_data = (
    no_outliers.groupby('state', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), sample_per_state), random_state=42))
    .reset_index(drop=True)
)

# Check the resulting dataset
print(f"Sampled data shape: {sampled_data.shape}")
print(sampled_data['state'].value_counts())  # Verify sample distribution across states
#%%
#############################################
# Step 2: Data Preprocessing
#############################################

# 2.1 Scale numerical features
numeric_features = ['price', 'house_size', 'bed', 'bath', 'acre_lot']
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(sampled_data[numeric_features])  # Normalize data

# 2.2 Encode categorical features (city, state)
categorical_features = ['city', 'state']
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(sampled_data[categorical_features]).toarray()

# 2.3 Combine scaled numerical and encoded categorical data
data_scaled_sampled = np.hstack((scaled_numeric, encoded_categorical))
print(f"Combined scaled data shape: {data_scaled_sampled.shape}")
#%%
#############################################
# Step 3: Determine Optimal Clusters using Elbow Method
#############################################
silhouette_scores = []
for k in range(2, 10):  # Test cluster numbers from 2 to 9
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_scaled_sampled)
    silhouette_scores.append(silhouette_score(data_scaled_sampled, labels))

# Plot silhouette scores to find the "elbow" point
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title("Elbow Method with Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Outcome: The optimal number of clusters is 3.
#%%
#############################################
# Step 4: Apply K-Means Clustering
#############################################
optimal_clusters = 3  # Based on Elbow Method
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
sampled_data['kmeans_cluster'] = kmeans.fit_predict(data_scaled_sampled)
#%%
#############################################
# Step 5: Composite Scoring
#############################################
# Combine scaled numeric and encoded categorical feature names
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
features = numeric_features + list(encoded_feature_names)

# Extract one-hot encoded features for state and city
state_features = [feature for feature in encoded_feature_names if feature.startswith('state_')]
city_features = [feature for feature in encoded_feature_names if feature.startswith('city_')]

# Create a DataFrame for cluster centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

# Assign weights for composite scoring
weights = {
    'price': 0.4,
    'house_size': 0.3,
    'bed': 0.1,
    'bath': 0.1,
    'acre_lot': 0.05,
    'state': 0.05,  # All state features combined
    'city': 0.05    # All city features combined
}

# Compute composite score for each cluster
centroids['composite_score'] = (
    centroids['price'] * weights['price'] +
    centroids['house_size'] * weights['house_size'] +
    centroids['bed'] * weights['bed'] +
    centroids['bath'] * weights['bath'] +
    centroids['acre_lot'] * weights['acre_lot'] +
    centroids[state_features].sum(axis=1) * weights['state'] +
    centroids[city_features].sum(axis=1) * weights['city']
)

# Sort centroids by composite score and assign categories
sorted_centroids = centroids.sort_values(by='composite_score').reset_index()
cluster_mapping = {sorted_centroids.iloc[i].name: label for i, label in enumerate(['Affordable', 'Mid-Range', 'High-End'])}
sampled_data['category'] = sampled_data['kmeans_cluster'].map(cluster_mapping)

# Outcome: Each property is categorized into Affordable, Mid-Range, or High-End.
#%%
#############################################
# Step 6: Visualization
#############################################
# Scatter plot of house size vs. price, colored by category
sns.scatterplot(
    x=sampled_data['house_size'],
    y=sampled_data['price'],
    hue=sampled_data['category'],
    palette='coolwarm',
    legend='full'
)
plt.title('K-Means Clustering on Sampled Data: Price vs House Size')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.legend(title='Category')
plt.show()

# Outcome: Visual confirmation of clustering with distinct categories.
#%%
#############################################
# Step 7: Evaluate Clustering Quality
#############################################
# Calculate silhouette score for the final clustering
silhouette_avg = silhouette_score(data_scaled_sampled, sampled_data['kmeans_cluster'])
print(f"Silhouette Score for K-Means Clustering (Sampled Data): {silhouette_avg:.2f}")

# Outcome: The silhouette score  0.20 indicates modest cluster quality.

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
