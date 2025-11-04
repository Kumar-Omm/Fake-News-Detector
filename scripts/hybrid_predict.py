from transformers import pipeline
from newspaper import Article
from serpapi import GoogleSearch
import validators
import os

# Load fine-tuned model
model_path = "../models/saved_model"
classifier = pipeline("text-classification", model=model_path)

# 🔑 Add your SerpAPI key here
SERPAPI_KEY = "YOUR_SERPAPI_KEY"

def extract_article_text(url):
    if not validators.url(url):
        return None, "Invalid URL."
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text, None
    except Exception as e:
        return None, str(e)

def verify_with_google(text):
    search = GoogleSearch({
        "q": text[:100],  # take first 100 chars as query
        "num": 5,
        "api_key": SERPAPI_KEY
    })
    results = search.get_dict().get("organic_results", [])
    sources = [r["link"] for r in results]
    return sources

def hybrid_prediction(url):
    print(f"🔍 Analyzing: {url}")
    text, err = extract_article_text(url)
    if err:
        return {"error": err}

    # Step 1: Model prediction
    pred = classifier(text[:2000])[0]
    label = pred["label"]
    score = round(pred["score"], 2)

    # Step 2: Google verification
    sources = verify_with_google(text)
    credibility = "High" if any("bbc" in s or "reuters" in s or "nytimes" in s for s in sources) else "Low"

    result = {
        "model_prediction": label,
        "confidence": score,
        "verified_sources": sources,
        "credibility": credibility
    }

    return result

if __name__ == "__main__":
    test_url = "https://example.com/some-news-article"
    print(hybrid_prediction(test_url))
