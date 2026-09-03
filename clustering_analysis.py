"""
Week 3 Task: Unsupervised Learning and Clustering Analysis
Dataset: Country Development Data (public dataset, "HELP International" / Kaggle)
Source:  https://huggingface.co/datasets/kheejay88/country_data

Context: HELP International is a fictional NGO with $10 million to allocate,
trying to identify which countries most need aid based on socio-economic
and health factors. Run this in Google Colab.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# ---------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ---------------------------------------------------------------
# Download country_data_real.csv from the files I gave you and upload it
# to your Colab session (folder icon on the left -> upload), OR use the
# huggingface link below if you'd rather fetch it directly.
df = pd.read_csv("country_data_real.csv")

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ---------------------------------------------------------------
# STEP 2: PREPROCESSING
# ---------------------------------------------------------------
# Clustering algorithms are distance-based, so features on very different
# scales (income in the tens of thousands vs. total_fer as a small decimal)
# would let the largest-scale feature dominate the distance calculation.
# Standardizing puts every feature on the same footing.
features = ["child_mort", "exports", "health", "imports", "income",
            "inflation", "life_expec", "total_fer", "gdpp"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("\nScaled feature means (should be ~0):\n", X_scaled.mean().round(2))
print("\nScaled feature std devs (should be ~1):\n", X_scaled.std().round(2))

# ---------------------------------------------------------------
# STEP 3: FINDING THE RIGHT NUMBER OF CLUSTERS
# ---------------------------------------------------------------
# Elbow method: plot inertia (within-cluster sum of squares) for a range
# of k, and look for the point where adding more clusters stops helping much.
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

print("\nInertia by k:", dict(zip(k_range, [round(i, 1) for i in inertias])))
print("Silhouette score by k:", dict(zip(k_range, [round(s, 3) for s in silhouette_scores])))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#2c7fb8")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="#d95f0e")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("01_elbow_and_silhouette.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# STEP 4: FIT K-MEANS WITH THE CHOSEN K
# ---------------------------------------------------------------
K = 3  # chosen from the elbow/silhouette plots above
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nCluster sizes (k={K}):")
print(df["cluster"].value_counts().sort_index())

print("\nCluster centers (mean of each feature per cluster, original scale):")
print(df.groupby("cluster")[features].mean().round(1))

# ---------------------------------------------------------------
# STEP 5: VISUALIZE CLUSTERS WITH PCA
# ---------------------------------------------------------------
# The data has 9 dimensions, which can't be plotted directly. PCA reduces
# it to 2 principal components that capture most of the variance, purely
# for visualization purposes — the clustering itself was done on all 9.
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["pca1"] = pca_result[:, 0]
df["pca2"] = pca_result[:, 1]

print(f"\nVariance explained by 2 PCA components: {pca.explained_variance_ratio_.sum():.1%}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="Set1", s=60)
plt.title(f"Country Clusters (K-Means, k={K}) — PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("02_pca_clusters.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# STEP 6: CHARACTERIZE EACH CLUSTER
# ---------------------------------------------------------------
cluster_summary = df.groupby("cluster")[["child_mort", "income", "life_expec", "gdpp"]].mean().round(1)
print("\nKey indicators by cluster:")
print(cluster_summary)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, ["child_mort", "income", "life_expec", "gdpp"]):
    sns.boxplot(data=df, x="cluster", y=col, hue="cluster", palette="Set1", legend=False, ax=ax)
    ax.set_title(col)
plt.suptitle("Distribution of Key Indicators by Cluster")
plt.tight_layout()
plt.savefig("03_cluster_boxplots.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# STEP 7: WHICH COUNTRIES NEED AID MOST?
# ---------------------------------------------------------------
# Rank clusters by a combined "need" signal — high child mortality and
# low income/gdpp/life expectancy indicate the countries most in need of aid.
need_ranking = df.groupby("cluster")[["child_mort", "income", "gdpp", "life_expec"]].mean()
neediest_cluster = need_ranking.sort_values(
    ["child_mort", "income"], ascending=[False, True]
).index[0]

print(f"\nCluster most in need of aid (highest child mortality, lowest income): Cluster {neediest_cluster}")
print("\nSample countries in that cluster:")
print(df[df["cluster"] == neediest_cluster]["country"].head(15).tolist())

df.to_csv("country_data_clustered.csv", index=False)
print("\nSaved clustered dataset to country_data_clustered.csv")
