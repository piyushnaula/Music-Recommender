import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


AUDIO_FEATURES = [
    "acousticness", "danceability", "energy",
    "instrumentalness", "liveness", "speechiness",
    "tempo", "valence",
]

DATA_PATH = os.path.join("data", "processed.parquet")
MODELS_DIR = "models"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading processed.parquet ...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run src/preprocess.py first.")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows.")

    missing = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in parquet: {missing}")
        sys.exit(1)

    feature_df = df[AUDIO_FEATURES].copy()

    feature_df = feature_df.fillna(0.0)

    print("Fitting MinMaxScaler ...")
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(feature_df)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler → {scaler_path}")

    print("Training KNN model (n_neighbors=20, metric=cosine) ...")
    knn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(feature_matrix)

    knn_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    joblib.dump(knn, knn_path)
    print(f"  Saved KNN model → {knn_path}")

    print("Training KMeans model (n_clusters=30) — this may take 1–2 minutes ...")
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    df["cluster_id"] = cluster_labels

    df.to_parquet(DATA_PATH, engine="pyarrow", index=False)
    print(f"  Updated processed.parquet with cluster_id column.")

    kmeans_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    joblib.dump(kmeans, kmeans_path)
    print(f"  Saved KMeans model → {kmeans_path}")

    matrix_path = os.path.join(MODELS_DIR, "feature_matrix.npy")
    np.save(matrix_path, feature_matrix)
    print(f"  Saved feature matrix → {matrix_path}")

    print("\nAll models trained and saved successfully!")
    print(f"  scaler.pkl, knn_model.pkl, kmeans_model.pkl, feature_matrix.npy → {MODELS_DIR}/")


if __name__ == "__main__":
    main()
