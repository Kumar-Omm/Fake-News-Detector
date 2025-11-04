import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def merge_datasets(fake_path, true_path, save_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    df = df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Merged dataset saved to {save_path} with {len(df)} samples.")

if __name__ == "__main__":
    merge_datasets("data/fake.csv", "data/true.csv", "data/merged.csv")
