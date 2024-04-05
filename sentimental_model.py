from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('/home/dm-potato/data/dm-eunsu/capstone_code/sent_anal/train/results_ver1/checkpoint-8572')

# 새로운 입력 데이터
new_inputs = ["there is a bicycle parked on the sidewalk next to a building.", "there is a man that is standing on the street with a bat."]

# 입력 데이터를 모델이 이해할 수 있는 형태로 전처리
encoded_inputs = tokenizer(new_inputs, padding=True, truncation=True, return_tensors="pt")

# 모델을 평가 모드로 설정
model.eval()

# 추론 실행
with torch.no_grad():
    outputs = model(**encoded_inputs)

# 로짓 추출 및 소프트맥스 적용
logits = outputs.logits
predictions = torch.softmax(logits, dim=-1)

# 결과 해석
for i, input_text in enumerate(new_inputs):
    print(f"Input: {input_text}")
    print(f"Predictions: {predictions[i].tolist()}")
