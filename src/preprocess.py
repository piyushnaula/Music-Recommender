import pandas as pd
import os
import sys


def main():
    raw_csv = os.path.join("data", "spotify_tracks.csv")
    output_parquet = os.path.join("data", "processed.parquet")

    if not os.path.exists(raw_csv):
        print(f"ERROR: {raw_csv} not found. Place spotify_tracks.csv inside the data/ folder.")
        sys.exit(1)

    df = pd.read_csv(raw_csv, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    clip_cols = [
        "acousticness", "danceability", "energy",
        "instrumentalness", "liveness", "speechiness", "valence",
    ]
    clip_cols = [c for c in clip_cols if c in df.columns]
    df[clip_cols] = df[clip_cols].clip(lower=0.0, upper=1.0)
    print(f"  Clipped {len(clip_cols)} audio feature columns to [0, 1].")
    if "language" in df.columns:
        before = (df["language"] == "Unknown").sum()
        df["language"] = df["language"].replace("Unknown", "Other")
        print(f"  Replaced {before:,} 'Unknown' language values with 'Other'.")
    else:
        print("  WARNING: 'language' column not found – skipping language fix.")

    if "duration_ms" in df.columns:
        df["duration_min"] = (df["duration_ms"] / 60000).round(2)
        print("  Added duration_min column.")
    else:
        print("  WARNING: 'duration_ms' column not found – skipping duration_min.")

    if "track_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="track_id", keep="first")
        dropped = before - len(df)
        print(f"  Dropped {dropped:,} duplicate track_id rows. {len(df):,} rows remain.")
    else:
        print("  WARNING: 'track_id' column not found – skipping deduplication.")

    os.makedirs("data", exist_ok=True)
    df.to_parquet(output_parquet, engine="pyarrow", index=False)
    print(f"\nSaved cleaned data → {output_parquet}")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
