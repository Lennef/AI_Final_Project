from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
import numpy as np
import evaluate

# 1. 載入資料
dataset=load_from_disk("super-emotion")

# 2. 用好標籤
labels_list = dataset['train'].features['labels']
print(type(labels_list)) 
