import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from newspaper import Article
import validators

# ==============================
# 🧠 Load Model + Tokenizer
# ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ==============================
# 📰 Function: Extract text from URL
# ==============================
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return None

# ==============================
# 🧾 Function: Predict Fake/Real
# ==============================
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    label = "📰 Real News" if probs[1] > probs[0] else "⚠️ Fake News"
    confidence = round(float(probs.max()) * 100, 2)
    return label, confidence

# ==============================
# 🚀 Main Test
# ==============================
if __name__ == "__main__":
    while True:
        user_input = input("\nPaste a news URL or text (or 'exit' to quit):\n> ")

        if user_input.lower() == "exit":
            print("👋 Exiting...")
            break

        if validators.url(user_input):
            print("🔗 Extracting article text...")
            text = extract_text_from_url(user_input)
        else:
            text = user_input

        if text:
            label, conf = predict_news(text)
            print(f"\n✅ Prediction: {label} ({conf}% confidence)\n")
        else:
            print("⚠️ Unable to extract or process text.")
