# Model Card for Customer Segmentation Model

## Model Overview
- **Task**: Customer Segmentation
- **Algorithm**: KMeans Clustering (Best Model)
- **Features**: 
  - **Recency**: How recently a customer has made a purchase, measured in days.
  - **Frequency**: The total number of purchases made by the customer during a specific period.
  - **Monetary**: The total amount of money spent by the customer.
  - **AvgOrderValue**: The average value of each order placed by the customer.
  - **UniqueProducts**: The number of unique products purchased by the customer.
- **Preprocessing**: Features were scaled prior to clustering to standardize the input for the KMeans algorithm.

## Intended Use and Out-of-Scope Uses
- **Intended Use**: This model is intended for use in the retail industry to segment customers based on purchasing behavior, enabling targeted marketing strategies and personalized communications.
- **Out-of-Scope Uses**: This model should not be used for predicting individual customer spending or for any application outside of customer segmentation analysis.

## Training Data Description
- **Dataset**: The model was trained on a dataset containing 5,878 customers with the following features derived from their purchasing behavior: Recency, Frequency, Monetary, AvgOrderValue, and UniqueProducts. The dataset is named features.csv.

## Feature Descriptions
- **Recency**: A continuous value representing the number of days since the last purchase.
- **Frequency**: A continuous value indicating the number of purchases made by the customer within the observation period.
- **Monetary**: A continuous value representing the total amount spent by the customer.
- **AvgOrderValue**: A continuous value representing the average order value of the customer's transactions.
- **UniqueProducts**: A continuous value indicating the number of distinct products the customer has purchased.

## Performance Metrics
- **Best Model**: KMeans
  - **Silhouette Score**: 0.4383
  - **Inertia**: 12,576.99
- **Cluster Sizes**:
  - Cluster 0: 3,734 customers
  - Cluster 1: 14 customers
  - Cluster 2: 2,129 customers
  - Cluster 3: 1 customer

## Ethical Considerations
- The model may inadvertently reinforce biases if the training data reflects historical inequities. Consideration should be given to the demographic representation and fairness of segmentation outcomes to avoid exclusionary practices.

## Limitations
- The KMeans algorithm requires a predetermined number of clusters; thus, the choice of 4 clusters may not be optimal for all datasets. Additionally, the model may not capture the underlying customer behavior complexities.
- Clusters may contain very few customers (e.g., Cluster 1 has only 14 customers), leading to potential interpretability issues.

## Usage Example
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('path/to/features.csv')

# Preprocessing: Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Instantiate and fit the KMeans model
kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_features)

# Predict clusters
data['Cluster'] = kmeans.predict(scaled_features)
print(data[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'Cluster']])
```