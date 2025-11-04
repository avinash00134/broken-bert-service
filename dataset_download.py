from datasets import load_dataset
import pandas as pd

# Number of samples you want to extract
n_samples = 20000  # You can adjust to 5000, 20000, etc.

# Load Amazon polarity dataset (doesn't download full file at once)
dataset = load_dataset("amazon_polarity", split=f"train[:{n_samples}]")

# Convert to pandas and format
df = dataset.to_pandas()[["content", "label"]]
df.columns = ["review", "label"]
df["label"] = df["label"].map({0: "negative", 1: "positive"})

# Save to CSV
df.to_csv("assets/reviews.csv", index=False)

print(f"Saved {len(df)} samples to assets/reviews.csv")
