# Evaluation Report — Customer Segmentation

Generated: 2026-03-10 23:36:57

## Dataset
- Source: `outputs/features.csv`
- Customers: 5,878
- Features: ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts']
- n_clusters: 4

## Model Comparison

| Model | Silhouette Score | Inertia |
|---|---|---|
| KMeans (n_clusters=4, random_state=42) | 0.4383 | 12,576.99 |
| AgglomerativeClustering (n_clusters=4) | 0.4183 | N/A |

## Best Model: KMeans
- Silhouette Score: 0.4383
- Model saved to: `outputs/model.pkl`

## Cluster Sizes

### KMeans
- 0: 3734 customers
- 1: 14 customers
- 2: 2129 customers
- 3: 1 customers

### AgglomerativeClustering
- 0: 4185 customers
- 1: 8 customers
- 2: 1684 customers
- 3: 1 customers

## Best Model Cluster Profiles (scaled features)
  Cluster 0 (n=3,734): Recency=-0.668, Frequency=0.134, Monetary=0.032, AvgOrderValue=0.004, UniqueProducts=0.205
  Cluster 1 (n=14): Recency=-0.945, Frequency=13.041, Monetary=14.264, AvgOrderValue=1.044, UniqueProducts=9.060
  Cluster 2 (n=2,129): Recency=1.179, Frequency=-0.321, Monetary=-0.156, AvgOrderValue=-0.046, UniqueProducts=-0.420
  Cluster 3 (n=1): Recency=-0.957, Frequency=-0.330, Monetary=11.463, AvgOrderValue=69.060, UniqueProducts=-0.678

## Notes
- Features were standardized with StandardScaler before training.
- Silhouette score ranges from -1 (poor) to 1 (perfect); higher is better.
- Inertia (KMeans only): sum of squared distances of samples to their nearest cluster centre.
