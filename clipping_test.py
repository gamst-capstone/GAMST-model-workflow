from capstone_function import loadCaptionModel, loadDetectionModel, loadSentimentalModel, generate_sentimental_score
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

import cv2, os, torch

processor= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

yolo_model = None

sentimental_model = AutoModelForSequenceClassification.from_pretrained('/home/hyeok/GAMST-model-workflow/sentimental_model')
sentimental_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sentimental_model.eval()

def generate_sentimental_score(caption_sentence):
    # 입력 데이터를 모델이 이해할 수 있는 형태로 전처리
    encoded_inputs = sentimental_tokenizer(caption_sentence, padding=True, truncation=True, return_tensors="pt")
    
    # 모델에 입력값을 통해 출력값 계산
    with torch.no_grad():
        outputs = sentimental_model(**encoded_inputs)

    # 로짓 추출 및 소프트맥스 적용
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1)

    # 결과 해석
    print(f"Input: {caption_sentence}")
    print(f"Predictions: {predictions[0].tolist()}")

    return predictions[0].tolist()


def clip_and_save(out, filename):
    out.release()

    # check the video file has been made
    if os.path.exists(filename):
        print(f"Video file {filename} has been made.")


# Sentimental score threshold (부정값)
SENTIMENTAL_THRESHOLD = 0.85

cap = cv2.VideoCapture("http://192.168.0.95:8000/stream.mjpg")
out = None

frame_count = 0
risk_section = []
is_N = False
start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_time = datetime.now()
    if not ret:
        break

    if frame_count % 10 == 0:
        crop_data_caption = None
        pil_raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # raw_input = processor(pil_raw_image, return_tensors="pt")
        
        raw_input = processor(pil_raw_image, return_tensors="pt").to("cuda")
        raw_out = blip_model.generate(**raw_input)

        original_data_caption = (processor.decode(raw_out[0], skip_special_token=True))
        original_data_caption = original_data_caption.replace(' [SEP]', '.')
        print(f"Frame Count: {frame_count}, Original_data_caption: {original_data_caption}")

        # 추출된 Caption Sentence를 바탕으로 Sentimental Model 적용
        predicted_risk = generate_sentimental_score(original_data_caption)

        # Decision RISK
        is_risky = 'N' if predicted_risk[1] >= SENTIMENTAL_THRESHOLD else 'P'
        is_risky_crop = None


        # Section detection
        if is_risky == 'N':
            start_time = frame_time

            # frame write here
            if out is None:
                filename = f"output_{start_time.strftime('%y%m%d%H%M%S')}.avi"
                out = cv2.VideoWriter(filename ,cv2.VideoWriter_fourcc(*'XVID'), 20, (640,480))
            out.write(frame)
            
            if is_N == False:
                is_N = True
        elif is_risky == 'P':
            if is_N == True:
                # frame write here
                out.write(frame)

                if (frame_time-start_time).seconds > 10:
                    risk_section.append([start_time, frame_time])
                    print(f"Risk Section: {risk_section}")
                    
                    # risk section video 추출
                    clip_and_save(out, filename)
                    out = None

                    is_N = False
                    start_time = 0

    frame_count += 1