MOOD_THRESHOLDS: dict[str, dict[str, tuple]] = {
    "Happy": {
        "valence": (0.65, None),
        "energy": (0.55, None),
    },
    "Sad": {
        "valence": (None, 0.35),
        "energy": (None, 0.40),
    },
    "Party": {
        "valence": (0.60, None),
        "energy": (0.70, None),
        "danceability": (0.65, None),
    },
    "Chill": {
        "valence": (0.30, 0.65),
        "energy": (None, 0.45),
        "acousticness": (0.40, None),
    },
    "Workout": {
        "energy": (0.80, None),
        "tempo": (120, None),
    },
    "Study": {
        "energy": (None, 0.40),
        "speechiness": (None, 0.10),
        "instrumentalness": (0.20, None),
    },
    "Romantic": {
        "valence": (0.55, None),
        "energy": (0.25, 0.60),
        "acousticness": (0.40, None),
    },
}

ACTIVITY_THRESHOLDS: dict[str, dict[str, tuple]] = {
    "Gym": {
        "energy": (0.80, None),
        "tempo": (125, None),
        "danceability": (0.60, None),
    },
    "Sleep": {
        "energy": (None, 0.25),
        "acousticness": (0.60, None),
        "instrumentalness": (0.30, None),
    },
    "Drive": {
        "energy": (0.50, 0.80),
        "valence": (0.45, None),
    },
    "Cook": {
        "danceability": (0.65, None),
        "valence": (0.55, None),
        "energy": (0.40, 0.75),
    },
    "Focus": {
        "energy": (None, 0.45),
        "speechiness": (None, 0.10),
        "instrumentalness": (0.15, None),
        "acousticness": (0.30, None),
    },
}

MOOD_DESCRIPTIONS: dict[str, str] = {
    "Happy": "Uplifting songs to brighten your day",
    "Sad": "Gentle songs for quiet, reflective moments",
    "Party": "High-energy tracks to get the crowd going",
    "Chill": "Relaxed vibes for winding down",
    "Workout": "Fast and powerful tracks to push your limits",
    "Study": "Calm, focused music with minimal lyrics",
    "Romantic": "Warm, acoustic songs for special moments",
}

ACTIVITY_EMOJIS: dict[str, str] = {
    "Gym": "🏋️",
    "Sleep": "😴",
    "Drive": "🚗",
    "Cook": "🍳",
    "Focus": "📚",
}

MOOD_EMOJIS: dict[str, str] = {
    "Happy": "😊",
    "Sad": "😢",
    "Party": "🎉",
    "Chill": "😌",
    "Workout": "💪",
    "Study": "📖",
    "Romantic": "❤️",
}

LANGUAGE_LABELS: dict[str, str] = {
    "Unknown": "Other",
    "English": "English",
    "Tamil": "Tamil",
    "Hindi": "Hindi",
    "Korean": "Korean",
    "Telugu": "Telugu",
    "Malayalam": "Malayalam",
    "Other": "Other",
}

LANGUAGE_OPTIONS = ["All", "English", "Tamil", "Hindi", "Korean", "Telugu", "Malayalam", "Other"]
