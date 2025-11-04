import os
import torch
import argparse
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt


# ==============================
# 🔧 Argument Parser
# ==============================
parser = argparse.ArgumentParser(description="Train or resume fake news detection model")
parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
args = parser.parse_args()


# ==============================
# 🗂 Setup & Folder Safety
# ==============================
base_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
model_dir = os.path.join(base_dir, "../models/saved_model")
os.makedirs(model_dir, exist_ok=True)


# ==============================
# 📚 Load and Prepare Data
# ==============================
print("📚 Loading and preparing data...")

fake_df = pd.read_csv("data/fake.csv")
true_df = pd.read_csv("data/true.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42)
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)


# ==============================
# ✂️ Tokenization
# ==============================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)


# ==============================
# 📏 Metrics
# ==============================
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


# ==============================
# 🧠 Model
# ==============================
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# ==============================
# ⚙️ Training Arguments
# ==============================
args_train = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    eval_strategy="steps",          # Evaluate every X steps
    save_strategy="steps",          # Save checkpoint every X steps
    save_steps=500,                 # Save every 500 steps
    eval_steps=500,
    logging_strategy="steps",
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=0,
    weight_decay=0.01,
    report_to="none",
    disable_tqdm=False,
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    save_total_limit=3,             # Keep only 3 latest checkpoints
)


# ==============================
# 🧑‍🏫 Trainer
# ==============================
trainer = Trainer(
    model=model,
    args=args_train,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# ==============================
# 🔄 Auto-Resume from Checkpoint
# ==============================
last_checkpoint = None
if args.resume:
    checkpoints = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"🔄 Resuming from checkpoint: {last_checkpoint}")
    else:
        print("⚠️ No checkpoints found. Starting fresh.")
else:
    print("🆕 Starting a fresh training run (ignoring old checkpoints).")


# ==============================
# 🚀 Train
# ==============================
print("🚀 Starting training (saving every 500 steps)...")
train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model(os.path.join(base_dir, "../models/final_model"))
print("✅ Final model saved!")


# ==============================
# 📊 Evaluate
# ==============================
metrics = trainer.evaluate()
print("📊 Final Evaluation:", metrics)


# ==============================
# 📈 Plot Loss
# ==============================
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
steps = list(range(len(train_loss)))

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Loss")
if eval_loss:
    plt.plot(eval_loss, label="Validation Loss", linestyle="--")
plt.xlabel("Steps/Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "../models/training_curve_checkpointed.png"))
plt.show()

print("\n✅ Training complete! Charts saved at '../models/training_curve_checkpointed.png'")