import datasets
import numpy as np
import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import string
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report,accuracy_score
from collections import Counter
# 下載 nltk 資源（只需一次）
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')
#Load the dataset
dataset = datasets.load_from_disk("super-emotion")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

texts_train  = train_dataset["text"]
labels_train = train_dataset["labels"]

texts_test = test_dataset["text"]
labels_test = test_dataset["labels"]
# NLTK prerocessing
stop_words = set(stopwords.words("english"))
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # 移除標點、數字
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)
def preprocess_and_save(texts, output_file, batch_size=1000):
    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            cleaned_batch = [preprocess_text(t) for t in batch]
            for line in cleaned_batch:
                f.write(json.dumps(line) + "\n")
preprocess_and_save(texts_train, "texts_train_cleaned.jsonl")
preprocess_and_save(texts_test, "texts_test_cleaned.jsonl")
with open("labels_train.json", "w", encoding="utf-8") as f:
    json.dump(labels_train, f)
with open("labels_test.json", "w", encoding="utf-8") as f:
    json.dump(labels_test, f)
def load_cleaned_texts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
texts_cleaned_train = load_cleaned_texts("texts_train_cleaned.jsonl")
texts_cleaned_test = load_cleaned_texts("texts_test_cleaned.jsonl")
#RAM會爆炸!!
#texts_cleaned_train = [preprocess_text(t) for t in texts_train [:50000]]
#texts_cleaned_test = [preprocess_text(t) for t in texts_test [:50000]]
#labels_subset_train = labels_train[:50000]
#labels_subset_test = labels_test[:50000]
# TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(texts_cleaned_train)
def vectorize_in_batch(texts, batch_size=1000):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        X_batch = vectorizer.transform(batch).toarray().astype(np.float32)
        vectors.append(torch.tensor(X_batch))
    return vectors
# X = vectorizer.fit_transform(texts_cleaned_train).toarray().astype(np.float32)
X_train_batches = vectorize_in_batch(texts_cleaned_train)
X_test_batches = vectorize_in_batch(texts_cleaned_test)
# Label multi-hot encoding
#mlb = MultiLabelBinarizer()
# Y = mlb.fit_transform(labels_subset_train)   # shape = (n_samples, n_emotions)
#y_train = mlb.fit_transform(labels_subset_train)
#y_test = mlb.transform(labels_subset_test)
#y_train = mlb.fit_transform(labels_train)   # shape = (n_samples, n_emotions)
#y_test = mlb.transform(labels_test)

# 分割訓練集與測試集
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#X_train = torch.tensor(X_train, dtype=torch.float32)
#X_test = torch.tensor(X_test, dtype=torch.float32)
#原本為多標籤，取第一個標籤作為分類
y_train_single = [labels[0] for labels in labels_train]
y_test_single = [labels[0] for labels in labels_test]
#轉成tensor
train_data = [(batch, torch.tensor(y_train_single[i:i+len(batch)], dtype=torch.long)) for i, batch in enumerate(X_train_batches)]
test_data = [(batch, torch.tensor(y_test_single[i:i+len(batch)], dtype=torch.long)) for i, batch in enumerate(X_test_batches)]
#y_train = torch.tensor(y_train_single, dtype=torch.long)
#y_test = torch.tensor(y_test_single, dtype=torch.long)
#建立模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)
    
#設定類別的權重，因為資料集不平衡
label_counter = Counter(y_train_single)
class_counts = torch.tensor([label_counter[i] for i in range(7)], dtype=torch.float32)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()*7
#設定參數
input_dim = X_train_batches[0].shape[1]
hidden_dim = 256
output_dim = 7
#初始化模型
model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.002)
#訓練模型
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in train_data:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_data):.4f}, Acc: {acc:.4f}")
    model.eval()
total_loss = 0.0
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_data:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(y_batch)
    avg_loss = total_loss / len(test_data)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")