import sys
from contextlib import asynccontextmanager
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.recommender as recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI started. Models already loaded via recommender.py import.")
    yield
    print("FastAPI shutting down.")


app = FastAPI(
    title="Music Recommendation API",
    description="Backend for the Spotify music recommendation app.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health check")
def health():
    """Returns API status and total song count."""
    total = len(recommender.df)
    return {"status": "ok", "total_songs": total}


@app.get("/search", summary="Search songs by name, artist, or album")
def search(
    q: str = Query(..., description="Search term"),
    n: int = Query(15, ge=1, le=50, description="Number of results"),
):
    results = recommender.search_songs(query=q, n=n)
    return {"results": results, "count": len(results)}


@app.get("/recommend", summary="Get songs similar to a given track")
def recommend(
    track_id: str = Query(..., description="Spotify track ID"),
    n: int = Query(10, ge=1, le=50, description="Number of results"),
    language: Optional[str] = Query(None, description="Filter by language"),
):
    data = recommender.get_recommendations(track_id=track_id, n=n, language=language)
    if not data["results"]:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found in dataset.")
    return data


@app.get("/mood", summary="Get songs matching a mood")
def mood(
    mood: str = Query(..., description="Mood name: Happy, Sad, Party, Chill, Workout, Study, Romantic"),
    language: Optional[str] = Query(None, description="Filter by language"),
    n: int = Query(20, ge=1, le=100, description="Number of results"),
):
    valid_moods = list(recommender.MOOD_THRESHOLDS.keys()) if hasattr(recommender, "MOOD_THRESHOLDS") else []
    from src.mood_engine import MOOD_THRESHOLDS
    if mood not in MOOD_THRESHOLDS:
        raise HTTPException(status_code=400, detail=f"Invalid mood '{mood}'. Valid: {list(MOOD_THRESHOLDS.keys())}")
    data = recommender.get_mood_songs(mood=mood, language=language, n=n)
    return data


@app.get("/activity", summary="Get songs matching an activity")
def activity(
    activity: str = Query(..., description="Activity: Gym, Sleep, Drive, Cook, Focus"),
    language: Optional[str] = Query(None, description="Filter by language"),
    n: int = Query(20, ge=1, le=100, description="Number of results"),
):
    from src.mood_engine import ACTIVITY_THRESHOLDS
    if activity not in ACTIVITY_THRESHOLDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid activity '{activity}'. Valid: {list(ACTIVITY_THRESHOLDS.keys())}",
        )
    data = recommender.get_activity_songs(activity=activity, language=language, n=n)
    return data


@app.get("/trending", summary="Get trending songs sorted by popularity")
def trending(
    language: Optional[str] = Query(None, description="Filter by language"),
    year: Optional[int] = Query(None, ge=1971, le=2024, description="Filter by year"),
    n: int = Query(20, ge=1, le=100, description="Number of results"),
):
    data = recommender.get_trending(language=language, year=year, n=n)
    return data


@app.get("/song/{track_id}", summary="Get full details for a single song")
def song_details(track_id: str):
    details = recommender.get_song_details(track_id=track_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")
    return details


@app.get("/artist", summary="Get songs by an artist and similar artists")
def artist(
    name: str = Query(..., description="Artist name (partial match supported)"),
):
    songs = recommender.get_artist_songs(artist_name=name, n=10)
    similar = recommender.get_similar_artists(artist_name=name, n=5)
    return {"songs": songs, "similar_artists": similar}


@app.get("/playlist/chain", summary="Generate a playlist chain")
def playlist_chain(
    track_id: str = Query(..., description="Starting Spotify track ID"),
    length: int = Query(10, ge=2, le=20, description="Number of songs in playlist"),
):
    playlist = recommender.get_playlist_chain(start_track_id=track_id, length=length)
    if not playlist:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found in dataset.")
    return {"playlist": playlist, "count": len(playlist)}


@app.get("/cluster/{cluster_id}", summary="Get songs in a KMeans cluster")
def cluster(cluster_id: int):
    if cluster_id < 0 or cluster_id > 29:
        raise HTTPException(status_code=400, detail="cluster_id must be between 0 and 29.")
    songs = recommender.get_cluster_songs(cluster_id=cluster_id, n=10)
    return {"cluster_id": cluster_id, "songs": songs}


@app.get("/cluster/{cluster_id}/scatter", summary="Get all songs for cluster scatter plot")
def cluster_scatter(cluster_id: int):
    if cluster_id < 0 or cluster_id > 29:
        raise HTTPException(status_code=400, detail="cluster_id must be between 0 and 29.")
    all_songs = recommender.get_all_songs_for_scatter()
    return {"data": all_songs, "selected_cluster": cluster_id}
