import json
from typing import Optional
import sys

import os
import requests
import streamlit as st
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mood_engine import (
    MOOD_EMOJIS,
    MOOD_DESCRIPTIONS,
    ACTIVITY_EMOJIS,
    LANGUAGE_OPTIONS,
)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .song-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        border: 1px solid #2d2d44;
    }
    .song-title { font-size: 15px; font-weight: bold; color: #e0e0ff; }
    .song-artist { font-size: 13px; color: #a0a0c0; }
    .pill {
        display: inline-block;
        background: #2d2d44;
        color: #b0b0d0;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_api_status() -> dict:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "offline", "total_songs": 0}


def api_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        st.warning(f"API returned {resp.status_code}: {resp.text[:200]}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def load_image_from_url(url: str | None):
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content))
    except Exception:
        pass
    return None


def show_placeholder(label: str = "🎵", size: int = 80):
    st.markdown(
        f'<div style="width:{size}px;height:{size}px;background:#2d2d44;border-radius:8px;'
        f'display:flex;align-items:center;justify-content:center;font-size:28px;">{label}</div>',
        unsafe_allow_html=True,
    )


def show_song_card(song: dict, small: bool = False):
    art_size = 60 if small else 100
    img = load_image_from_url(song.get("artwork_url"))

    with st.container():
        st.markdown('<div class="song-card">', unsafe_allow_html=True)

        col_art, col_info = st.columns([1, 3])
        with col_art:
            if img:
                st.image(img, width=art_size)
            else:
                show_placeholder(size=art_size)

        with col_info:
            title = song.get("track_name", "Unknown")
            artist = song.get("artist_name", "Unknown")
            year = song.get("year", "")
            lang = song.get("language", "")
            pop = song.get("popularity", 0) or 0
            track_url = song.get("track_url", "")

            st.markdown(f'<div class="song-title">{title[:50]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="song-artist">{artist}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="pill">{year}</span><span class="pill">{lang}</span>',
                unsafe_allow_html=True,
            )
            st.progress(int(pop) / 100)

            if track_url:
                st.markdown(
                    f'<a href="{track_url}" target="_blank" style="font-size:13px;color:#1DB954;">▶ Play on Spotify</a>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


def show_song_grid(songs: list[dict], cols: int = 3, small: bool = False):
    if not songs:
        st.info("No songs found.")
        return
    columns = st.columns(cols)
    for i, song in enumerate(songs):
        with columns[i % cols]:
            show_song_card(song, small=small)


def make_radar_chart(song: dict, title: str = "") -> go.Figure:
    """Build a plotly radar chart for a song's 8 audio features."""
    features = ["energy", "valence", "danceability", "acousticness",
                "speechiness", "instrumentalness", "liveness"]
    tempo_norm = min((song.get("tempo") or 0) / 220, 1.0)
    feature_labels = features + ["tempo (norm)"]

    values = [song.get(f) or 0 for f in features] + [tempo_norm]
    values_display = values + [values[0]]       
    labels_display = feature_labels + [feature_labels[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_display,
            theta=labels_display,
            fill="toself",
            line_color="#1DB954",
            fillcolor="rgba(29, 185, 84, 0.2)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title or song.get("track_name", ""),
        paper_bgcolor="#1e1e2e",
        font_color="#e0e0ff",
        height=350,
    )
    return fig


try:
    from streamlit_option_menu import option_menu

    with st.sidebar:
        st.markdown("## 🎵 Music Recommender")

        selected_page = option_menu(
            menu_title="Navigate",
            options=["Home", "Search", "Recommend", "Explore", "Artist", "Playlist"],
            icons=["house", "search", "music-note", "compass", "person", "collection"],
            default_index=0,
            styles={
                "container": {"background-color": "#1e1e2e"},
                "icon": {"color": "#1DB954"},
                "nav-link-selected": {"background-color": "#2d2d44"},
            },
        )

        st.divider()

        language_filter = st.selectbox(
            "🌐 Language Filter",
            LANGUAGE_OPTIONS,
            key="language_filter",
        )

        st.divider()

        status = get_api_status()
        if status.get("status") == "ok":
            st.success(f"✅ API Online · {status.get('total_songs', 0):,} songs")
        else:
            st.error("❌ API Offline — start FastAPI first")

except ImportError:
    with st.sidebar:
        st.markdown("## 🎵 Music Recommender")
        selected_page = st.radio(
            "Navigate",
            ["Home", "Search", "Recommend", "Explore", "Artist", "Playlist"],
        )
        language_filter = st.selectbox("🌐 Language Filter", LANGUAGE_OPTIONS, key="language_filter")
        status = get_api_status()
        if status.get("status") == "ok":
            st.success(f"✅ API Online · {status.get('total_songs', 0):,} songs")
        else:
            st.error("❌ API Offline")


lang = st.session_state.get("language_filter", "All")
lang_param = None if lang == "All" else lang



if selected_page == "Home":
    st.title("🎵 What's your mood today?")
    st.markdown("Pick a mood and get songs that match how you feel.")

    moods = list(MOOD_EMOJIS.keys())
    mood_cols = st.columns(4)

    if "selected_mood" not in st.session_state:
        st.session_state["selected_mood"] = None

    for i, mood in enumerate(moods):
        with mood_cols[i % 4]:
            emoji = MOOD_EMOJIS[mood]
            desc = MOOD_DESCRIPTIONS[mood]
            if st.button(f"{emoji} {mood}", use_container_width=True, help=desc):
                st.session_state["selected_mood"] = mood

    selected_mood = st.session_state.get("selected_mood")

    if selected_mood:
        st.subheader(f"{MOOD_EMOJIS[selected_mood]} {selected_mood} Songs")
        st.caption(MOOD_DESCRIPTIONS[selected_mood])
        params = {"mood": selected_mood, "n": 20}
        if lang_param:
            params["language"] = lang_param

        data = api_get("/mood", params)
        if data:
            if data.get("fallback"):
                st.info(f"Not enough {lang_param} songs matched. Showing from all languages.")
            show_song_grid(data.get("results", []))

    st.divider()
    st.subheader("🔥 Trending Now")
    params = {"n": 5}
    if lang_param:
        params["language"] = lang_param
    trending_data = api_get("/trending", params)
    if trending_data:
        show_song_grid(trending_data.get("results", []), cols=5, small=True)



elif selected_page == "Search":
    st.title("🔍 Search Songs")

    query = st.text_input("Type a song name, artist, or album and press Enter")

    if query:
        data = api_get("/search", {"q": query, "n": 15})
        results = data.get("results", []) if data else []

        if not results:
            st.warning("No songs found. Try a different name.")
        else:
            st.success(f"Found {len(results)} songs")
            cols = st.columns(3)
            for i, song in enumerate(results):
                with cols[i % 3]:
                    show_song_card(song)
                    if st.button("Select →", key=f"sel_{song.get('track_id')}_{i}"):
                        st.session_state["selected_track_id"] = song.get("track_id")
                        st.session_state["selected_track"] = song
                        st.success(f"Selected: {song.get('track_name')}. Go to Recommend page →")



elif selected_page == "Recommend":
    st.title("🎯 Song Recommendations")

    track_id = st.session_state.get("selected_track_id")
    selected_track = st.session_state.get("selected_track")

    if not track_id:
        st.info("Go to the Search page and pick a song first.")
    else:
        details = api_get(f"/song/{track_id}")
        if details:
            col_art, col_info = st.columns([1, 3])
            with col_art:
                img = load_image_from_url(details.get("artwork_url"))
                if img:
                    st.image(img, width=180)
                else:
                    show_placeholder(size=180)
            with col_info:
                st.markdown(f"### {details.get('track_name', '')}")
                st.markdown(f"**Artist:** {details.get('artist_name', '')}")
                st.markdown(f"**Album:** {details.get('album_name', '')}")
                st.markdown(
                    f"**Year:** {details.get('year', '')}  ·  "
                    f"**Language:** {details.get('language', '')}  ·  "
                    f"**Popularity:** {details.get('popularity', 0)}/100"
                )
                track_url = details.get("track_url", "")
                if track_url:
                    st.markdown(
                        f'<a href="{track_url}" target="_blank">▶ Open on Spotify</a>',
                        unsafe_allow_html=True,
                    )

            st.subheader("🕸️ Audio Fingerprint")
            fig = make_radar_chart(details, title=details.get("track_name", ""))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎵 Songs Similar to This")
        params = {"track_id": track_id, "n": 10}
        if lang_param:
            params["language"] = lang_param

        rec_data = api_get("/recommend", params)
        if rec_data:
            if rec_data.get("fallback"):
                st.info(f"Not enough {lang_param} songs matched. Showing from all languages.")
            show_song_grid(rec_data.get("results", []))

        st.divider()
        with st.expander("🆚 Compare Two Songs"):
            compare_query = st.text_input("Search for a second song to compare", key="compare_search")
            if compare_query:
                comp_data = api_get("/search", {"q": compare_query, "n": 6})
                comp_results = comp_data.get("results", []) if comp_data else []
                if comp_results:
                    comp_options = {f"{s.get('track_name')} — {s.get('artist_name')}": s for s in comp_results}
                    comp_choice = st.selectbox("Pick a song", list(comp_options.keys()), key="comp_choice")
                    comp_song = comp_options[comp_choice]

                    song1_details = details or {}
                    song2_details = comp_song

                    if song1_details and song2_details:
                        fig_compare = make_subplots(
                            rows=1, cols=2,
                            specs=[[{"type": "polar"}, {"type": "polar"}]],
                            subplot_titles=[
                                song1_details.get("track_name", "Song 1"),
                                song2_details.get("track_name", "Song 2"),
                            ],
                        )

                        for col_num, song_data in enumerate([song1_details, song2_details], start=1):
                            features = ["energy", "valence", "danceability",
                                        "acousticness", "speechiness", "instrumentalness", "liveness"]
                            tempo_norm = min((song_data.get("tempo") or 0) / 220, 1.0)
                            labels = features + ["tempo (norm)"]
                            values = [song_data.get(f) or 0 for f in features] + [tempo_norm]
                            values_closed = values + [values[0]]
                            labels_closed = labels + [labels[0]]

                            color = "#1DB954" if col_num == 1 else "#e05252"
                            fig_compare.add_trace(
                                go.Scatterpolar(
                                    r=values_closed,
                                    theta=labels_closed,
                                    fill="toself",
                                    line_color=color,
                                    fillcolor=color.replace("#", "rgba(").replace("1DB954", "29,185,84,0.2").replace("e05252", "224,82,82,0.2") + ")",
                                    name=song_data.get("track_name", ""),
                                ),
                                row=1, col=col_num,
                            )

                        fig_compare.update_layout(
                            paper_bgcolor="#1e1e2e",
                            font_color="#e0e0ff",
                            height=400,
                            showlegend=False,
                        )
                        fig_compare.update_polars(radialaxis=dict(visible=True, range=[0, 1]))
                        st.plotly_chart(fig_compare, use_container_width=True)



elif selected_page == "Explore":
    st.title("🧭 Explore Music")

    tab_trending, tab_activity, tab_clusters = st.tabs(["🔥 Trending", "🎯 Activity", "🔬 Clusters"])
    with tab_trending:
        st.subheader("Trending Songs")
        year_choice = st.slider("Filter by year", min_value=1971, max_value=2024, value=2020)

        params = {"year": year_choice, "n": 20}
        if lang_param:
            params["language"] = lang_param

        data = api_get("/trending", params)
        if data:
            if data.get("fallback"):
                st.info(f"Not enough {lang_param} songs. Showing from all languages.")

            results = data.get("results", [])
            if results:
                import plotly.express as px
                top10 = results[:10]
                names = [f"{s.get('track_name', '')[:25]}..." if len(s.get('track_name', '')) > 25
                         else s.get('track_name', '') for s in top10]
                pops = [s.get("popularity", 0) for s in top10]

                bar_fig = px.bar(
                    x=pops, y=names, orientation="h",
                    labels={"x": "Popularity", "y": "Song"},
                    color=pops, color_continuous_scale="Greens",
                    title=f"Top 10 Songs in {year_choice}",
                )
                bar_fig.update_layout(
                    paper_bgcolor="#1e1e2e",
                    plot_bgcolor="#1e1e2e",
                    font_color="#e0e0ff",
                    yaxis=dict(autorange="reversed"),
                    height=380,
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(bar_fig, use_container_width=True)

                show_song_grid(results)

    with tab_activity:
        st.subheader("Songs by Activity")

        activities = list(ACTIVITY_EMOJIS.keys())
        act_cols = st.columns(len(activities))

        if "selected_activity" not in st.session_state:
            st.session_state["selected_activity"] = None

        for i, act in enumerate(activities):
            with act_cols[i]:
                emoji = ACTIVITY_EMOJIS[act]
                if st.button(f"{emoji} {act}", use_container_width=True):
                    st.session_state["selected_activity"] = act

        selected_activity = st.session_state.get("selected_activity")
        if selected_activity:
            st.subheader(f"{ACTIVITY_EMOJIS[selected_activity]} {selected_activity} Playlist")
            params = {"activity": selected_activity, "n": 20}
            if lang_param:
                params["language"] = lang_param

            data = api_get("/activity", params)
            if data:
                if data.get("fallback"):
                    st.info(f"Not enough {lang_param} songs matched. Showing from all languages.")
                show_song_grid(data.get("results", []))

    with tab_clusters:
        st.subheader("Music Mood Groups")
        st.caption("These are 30 music mood groups discovered automatically by AI.")

        cluster_id = st.slider("Select Cluster (0–29)", min_value=0, max_value=29, value=0)

        cluster_data = api_get(f"/cluster/{cluster_id}")
        if cluster_data:
            show_song_grid(cluster_data.get("songs", []))

        scatter_data = api_get(f"/cluster/{cluster_id}/scatter")
        if scatter_data and scatter_data.get("data"):
            import plotly.express as px
            import pandas as pd

            scatter_df = pd.DataFrame(scatter_data["data"])
            if not scatter_df.empty and "valence" in scatter_df.columns and "energy" in scatter_df.columns:
                scatter_df["cluster_id"] = scatter_df["cluster_id"].astype(str)
                scatter_df["is_selected"] = scatter_df["cluster_id"] == str(cluster_id)

                fig_scatter = px.scatter(
                    scatter_df,
                    x="valence", y="energy",
                    color="cluster_id",
                    opacity=0.4,
                    title=f"All songs — Cluster {cluster_id} highlighted",
                    labels={"valence": "Valence (happiness)", "energy": "Energy"},
                    hover_data=["track_name", "artist_name"] if "track_name" in scatter_df.columns else None,
                )
                fig_scatter.update_traces(marker=dict(size=3))
                fig_scatter.update_layout(
                    paper_bgcolor="#1e1e2e",
                    plot_bgcolor="#1e1e2e",
                    font_color="#e0e0ff",
                    height=450,
                    showlegend=False,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)



elif selected_page == "Artist":
    st.title("🎤 Artist Explorer")

    artist_input = st.text_input("Enter an artist name")

    if artist_input:
        data = api_get("/artist", {"name": artist_input})
        if data:
            songs = data.get("songs", [])
            similar = data.get("similar_artists", [])

            col_songs, col_similar = st.columns([3, 1])

            with col_songs:
                st.subheader(f"Songs by '{artist_input}'")
                if songs:
                    feature_keys = ["energy", "valence", "danceability", "acousticness",
                                    "speechiness", "instrumentalness", "liveness"]
                    avg_features = {
                        f: sum(s.get(f) or 0 for s in songs) / len(songs)
                        for f in feature_keys
                    }
                    avg_features["track_name"] = f"Avg: {artist_input}"
                    avg_features["tempo"] = sum(s.get("tempo") or 0 for s in songs) / len(songs)

                    radar_fig = make_radar_chart(avg_features, title=f"{artist_input} — Audio Fingerprint")
                    st.plotly_chart(radar_fig, use_container_width=True)

                    show_song_grid(songs, cols=2)
                else:
                    st.warning(f"No songs found for '{artist_input}'.")

            with col_similar:
                st.subheader("Similar Artists")
                if similar:
                    for entry in similar:
                        name = entry.get("artist_name", "")
                        score = entry.get("similarity", 0)
                        st.markdown(f"**{name}**")
                        st.progress(float(score))
                        if st.button(f"Search {name}", key=f"artist_{name}"):
                            st.session_state["_artist_redirect"] = name
                            st.rerun()
                else:
                    st.info("No similar artists found.")

    if st.session_state.get("_artist_redirect"):
        st.text_input("Enter an artist name", value=st.session_state.pop("_artist_redirect"))



elif selected_page == "Playlist":
    st.title("📋 Playlist Builder")
    st.caption("Pick a starting song and we'll build a playlist where each song flows into the next.")

    playlist_query = st.text_input("Search for a starting song")

    if playlist_query:
        search_data = api_get("/search", {"q": playlist_query, "n": 6})
        search_results = search_data.get("results", []) if search_data else []

        if search_results:
            options = {f"{s.get('track_name')} — {s.get('artist_name')}": s for s in search_results}
            choice_label = st.selectbox("Pick your starting song", list(options.keys()))
            start_song = options[choice_label]

            playlist_length = st.slider("Playlist length", min_value=5, max_value=20, value=10)

            if st.button("🎵 Build Playlist"):
                with st.spinner("Building your playlist ..."):
                    pl_data = api_get("/playlist/chain", {
                        "track_id": start_song.get("track_id"),
                        "length": playlist_length,
                    })

                if pl_data:
                    playlist = pl_data.get("playlist", [])
                    st.success(f"Playlist ready! {len(playlist)} songs.")

                    for idx, song in enumerate(playlist, start=1):
                        row_cols = st.columns([0.3, 1, 4, 2, 1, 1])
                        with row_cols[0]:
                            st.markdown(f"**{idx}.**")
                        with row_cols[1]:
                            img = load_image_from_url(song.get("artwork_url"))
                            if img:
                                st.image(img, width=50)
                            else:
                                show_placeholder(size=50)
                        with row_cols[2]:
                            st.markdown(f"**{song.get('track_name', '')}**")
                            st.caption(song.get("artist_name", ""))
                        with row_cols[3]:
                            st.caption(f"{song.get('duration_min', 0):.2f} min")
                        with row_cols[4]:
                            track_url = song.get("track_url", "")
                            if track_url:
                                st.markdown(
                                    f'<a href="{track_url}" target="_blank">▶</a>',
                                    unsafe_allow_html=True,
                                )

                    st.divider()
                    playlist_json = json.dumps(playlist, indent=2, default=str)
                    st.download_button(
                        label="⬇️ Export as JSON",
                        data=playlist_json,
                        file_name="playlist.json",
                        mime="application/json",
                    )
        else:
            st.warning("No songs found. Try a different search term.")
