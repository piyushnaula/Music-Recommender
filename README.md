# Music Recommendation System

A full-featured music recommendation web app built with FastAPI + Streamlit.
Dataset: 62,317 Spotify tracks across 7 languages (1971–2024).

---

## Features

- Search for any song and get similar songs
- Pick a mood (Happy, Sad, Party, Chill, Workout, Study, Romantic)
- Filter by language (Tamil, Hindi, English, Korean, Telugu, Malayalam)
- Filter by decade / year range
- Pick an activity (Gym, Sleep, Drive, Cook, Focus) and get a playlist
- Audio feature radar chart for any song
- Compare two songs side by side
- Browse trending songs by language or year
- Generate a playlist chain (each next song is similar to the previous one)
- Click album artwork to open song directly on Spotify

---

## How to Run (Step by Step)

```bash
pip install -r requirements.txt
```

```bash
python src/preprocess.py
```

```bash
python src/train.py
```
Creates `models/knn_model.pkl`, `models/kmeans_model.pkl`, `models/scaler.pkl`, `models/feature_matrix.npy`.

```bash
uvicorn api.main:app --reload --port 8000
```
Keep this terminal open. API: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

```bash
streamlit run app/streamlit_app.py
```
App opens at http://localhost:8501.

---

## Project Structure

```
music-recommender/
│
├── data/
│   ├── spotify_tracks.csv         
│   └── processed.parquet         
│
├── models/
│   ├── knn_model.pkl               
│   ├── kmeans_model.pkl            
│   ├── scaler.pkl                  
│   └── feature_matrix.npy         
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── recommender.py
│   └── mood_engine.py
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── .env
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /search?q= | Search songs |
| GET | /recommend?track_id= | Similar songs |
| GET | /mood?mood= | Songs by mood |
| GET | /activity?activity= | Songs by activity |
| GET | /trending | Trending songs |
| GET | /song/{track_id} | Song details |
| GET | /artist?name= | Artist songs + similar artists |
| GET | /playlist/chain?track_id= | Playlist chain |
| GET | /cluster/{id} | Songs in a KMeans cluster |