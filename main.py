import torch
import torch.nn as nn
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from joblib import load

app = Flask(__name__)

# 定義模型架構
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 載入 TfidfVectorizer
vectorizer = load("tfidf_vectorizer.pkl")
input_dim = vectorizer.max_features  # 取向量維度
hidden_dim = 256  
output_dim = 7    

# 載入模型
model = MLP(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("mlp_weights.pth", map_location=torch.device('cpu')))
model.eval()

def preprocess_text(text):
    text = text.lower()
    tokens = [t for t in text.split() if t.isalpha()]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        user_input = request.form["text"]
        cleaned = preprocess_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray().astype('float32')
        input_tensor = torch.tensor(vector, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        result = f"模型預測結果: {pred}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
