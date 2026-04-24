import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mood_engine import MOOD_THRESHOLDS, ACTIVITY_THRESHOLDS

AUDIO_FEATURES = [
    "acousticness", "danceability", "energy",
    "instrumentalness", "liveness", "speechiness",
    "tempo", "valence",
]

RETURN_COLS = [
    "track_id", "track_name", "artist_name", "album_name",
    "year", "popularity", "language", "artwork_url", "track_url",
    "duration_min", "energy", "valence", "danceability",
    "acousticness", "tempo", "speechiness", "instrumentalness",
    "liveness", "cluster_id",
]

_DATA_PATH = os.path.join("data", "processed.parquet")
_MODELS_DIR = "models"


def _load_all():
    print("recommender.py: loading data and models ...")

    df = pd.read_parquet(_DATA_PATH)

    if "cluster_id" not in df.columns:
        df["cluster_id"] = -1

    if "duration_min" not in df.columns and "duration_ms" in df.columns:
        df["duration_min"] = (df["duration_ms"] / 60000).round(2)

    knn = joblib.load(os.path.join(_MODELS_DIR, "knn_model.pkl"))
    kmeans = joblib.load(os.path.join(_MODELS_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(_MODELS_DIR, "scaler.pkl"))
    feature_matrix = np.load(os.path.join(_MODELS_DIR, "feature_matrix.npy"))

    df = df.reset_index(drop=True)

    print(f"  Loaded {len(df):,} songs and all models.")
    return df, knn, kmeans, scaler, feature_matrix


df, knn_model, kmeans_model, scaler, feature_matrix = _load_all()


def _to_records(subset: pd.DataFrame) -> list[dict]:
    available = [c for c in RETURN_COLS if c in subset.columns]
    records = subset[available].copy()

    records = records.where(pd.notnull(records), None)
    return records.to_dict(orient="records")


def _apply_thresholds(source_df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=source_df.index)
    for feature, (low, high) in thresholds.items():
        if feature not in source_df.columns:
            continue
        if low is not None:
            mask &= source_df[feature] >= low
        if high is not None:
            mask &= source_df[feature] <= high
    return source_df[mask]

def _filter_language(source_df: pd.DataFrame, language: str | None) -> tuple[pd.DataFrame, bool]:
    if not language or language.lower() == "all":
        return source_df, False

    filtered = source_df[source_df["language"].str.lower() == language.lower()]
    if len(filtered) < 5:
        return source_df, True
    return filtered, False



def get_recommendations(track_id: str, n: int = 10, language: str | None = None) -> dict:
    matches = df.index[df["track_id"] == track_id].tolist()
    if not matches:
        return {"results": [], "fallback": False}

    row_idx = matches[0]
    query_vector = feature_matrix[row_idx].reshape(1, -1)

    distances, indices = knn_model.kneighbors(query_vector, n_neighbors=20)
    neighbour_indices = indices[0][1:]

    result_df = df.iloc[neighbour_indices].copy()
    result_df, fell_back = _filter_language(result_df, language)

    return {
        "results": _to_records(result_df.head(n)),
        "fallback": fell_back,
    }


def get_mood_songs(mood: str, language: str | None = None, n: int = 20) -> dict:
    thresholds = MOOD_THRESHOLDS.get(mood, {})
    filtered = _apply_thresholds(df, thresholds)
    filtered, fell_back = _filter_language(filtered, language)
    filtered = filtered.sort_values("popularity", ascending=False)
    return {"results": _to_records(filtered.head(n)), "fallback": fell_back}


def get_activity_songs(activity: str, language: str | None = None, n: int = 20) -> dict:
    thresholds = ACTIVITY_THRESHOLDS.get(activity, {})
    filtered = _apply_thresholds(df, thresholds)
    filtered, fell_back = _filter_language(filtered, language)
    filtered = filtered.sort_values("popularity", ascending=False)
    return {"results": _to_records(filtered.head(n)), "fallback": fell_back}


def get_trending(language: str | None = None, year: int | None = None, n: int = 20) -> dict:
    source = df.copy()
    if year is not None and "year" in source.columns:
        source = source[source["year"] == year]
    source, fell_back = _filter_language(source, language)
    source = source.sort_values("popularity", ascending=False)
    return {"results": _to_records(source.head(n)), "fallback": fell_back}


def search_songs(query: str, n: int = 15) -> list[dict]:
    q = query.strip().lower()
    mask = (
        df["track_name"].str.lower().str.contains(q, na=False)
        | df["artist_name"].str.lower().str.contains(q, na=False)
        | df["album_name"].str.lower().str.contains(q, na=False)
    )
    results = df[mask].sort_values("popularity", ascending=False).head(n)
    return _to_records(results)


def get_artist_songs(artist_name: str, n: int = 10) -> list[dict]:
    mask = df["artist_name"].str.lower().str.contains(artist_name.strip().lower(), na=False)
    results = df[mask].sort_values("popularity", ascending=False).head(n)
    return _to_records(results)


def get_similar_artists(artist_name: str, n: int = 5) -> list[dict]:
    feat_cols = [f for f in AUDIO_FEATURES if f in df.columns]

    artist_profiles = (
        df.groupby("artist_name")[feat_cols]
        .mean()
        .dropna()
    )

    if artist_name not in artist_profiles.index:
        matches = [a for a in artist_profiles.index if artist_name.lower() in a.lower()]
        if not matches:
            return []
        artist_name = matches[0]

    query_profile = artist_profiles.loc[[artist_name]].values 
    all_profiles = artist_profiles.values                      

    sims = cosine_similarity(query_profile, all_profiles)[0]    

    artist_list = artist_profiles.index.tolist()
    sim_pairs = sorted(
        [(artist_list[i], float(sims[i])) for i in range(len(artist_list)) if artist_list[i] != artist_name],
        key=lambda x: x[1],
        reverse=True,
    )

    return [{"artist_name": name, "similarity": round(score, 4)} for name, score in sim_pairs[:n]]


def get_playlist_chain(start_track_id: str, length: int = 10) -> list[dict]:
    playlist_ids: set[str] = {start_track_id}
    playlist_records: list[dict] = []

    start_matches = df.index[df["track_id"] == start_track_id].tolist()
    if not start_matches:
        return []

    start_details = get_song_details(start_track_id)
    if start_details:
        playlist_records.append(start_details)

    current_id = start_track_id

    while len(playlist_records) < length:
        recs = get_recommendations(current_id, n=5)
        found_next = False

        for song in recs["results"]:
            tid = song.get("track_id")
            if tid and tid not in playlist_ids:
                playlist_ids.add(tid)
                playlist_records.append(song)
                current_id = tid
                found_next = True
                break

        if not found_next:
            break

    return playlist_records


def get_song_details(track_id: str) -> dict | None:
    matches = df[df["track_id"] == track_id]
    if matches.empty:
        return None
    return _to_records(matches.head(1))[0]


def get_cluster_songs(cluster_id: int, n: int = 10) -> list[dict]:
    cluster_df = df[df["cluster_id"] == cluster_id]
    result = cluster_df.sort_values("popularity", ascending=False).head(n)
    return _to_records(result)


def get_all_songs_for_scatter() -> list[dict]:
    cols = [c for c in ["valence", "energy", "cluster_id", "track_name", "artist_name"] if c in df.columns]
    slim = df[cols].copy()
    slim = slim.where(pd.notnull(slim), None)
    return slim.to_dict(orient="records")
