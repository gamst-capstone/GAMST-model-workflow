# 라이브러리 호출
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

import cv2
import pymysql
import os, logging
import torch

from utils import video_id_check, camera_id_check, upload_to_s3, video_length_in_seconds


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "gamst")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD","")
SENTIMENTAL_PATH= os.getenv("SENTIMENTAL_PATH", "")
CAPTION_TABLE = os.getenv("CAPTION_TABLE", "video_caption")
RISK_TABLE = os.getenv("RISK_TABLE", "video_riskysection")
VIDEO_BUCKET = os.getenv("VIDEO_BUCKET", "")

processor = None
blip_model = None
yolo_model = None
sentimental_model = None
sentimental_tokenizer = None

# 추출 frame 단위
FRAME_CUTOFF = 10

# RISK Clip 최소시간, second
CLIP_TIME = 10

# Sentimental score threshold (부정값)
SENTIMENTAL_THRESHOLD = 0.85

def db_conn():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8'
        )
        
        if conn:
            logger.info("[*] DB Connected")
            return conn
        else:
            logger.error("[*] DB Connection Failed")
            return None
    except Exception as e:
        logger.error(e)
        return 

# Captioning Model 로드하는 함수
def loadCaptionModel():
    # BLIP Model Load
    logger.info("[*] Loading BLIP Model...")
    global processor, blip_model
    if processor is None or blip_model is None:
        processor= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        # blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


# YOLO Model 로드하는 함수
def loadDetectionModel():
    logger.info("[*] Loading YOLO Model...")
    global yolo_model
    if yolo_model is None:
        yolo_model= YOLO('yolov8n.pt')

# Sentimental Model 로드하는 함수
def loadSentimentalModel():
    logger.info("[*] Loading Sentimental Model...")
    global sentimental_model, sentimental_tokenizer
    if sentimental_model is None:
        sentimental_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENTAL_PATH)
        sentimental_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        sentimental_model.eval()

# input: VideoPath // output: (10프레임당 Crop된 image Caption 문장 + Original Image Caption 문장)
# if Detection Model에서 Person이 안잡힐 경우 --> Crop된 image Caption 문장: None
def generateCaption(input_object):
    x_vector = []
    y_vector = []
    names = yolo_model.names

    video_path = input_object.get('url')
    input_type = input_object.get('type')

    logger.info("[*] Loading Video File...")    
    # video_path = HTTP URL from s3 bucket object OR Camera URL 
    cap = cv2.VideoCapture(video_path)
    out = None
    frame_count = 0
    risk_section = []
    is_N = False
    start_time = 0

    if input_type == 'video':
        video_uid = input_object.get('video_uid')
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_clip_num = 0
        logger.info(f"[*] Video Length: {length}, FPS: {fps}")
    elif input_type == 'stream':
        camera_id = input_object.get('id')
        start_time = input_object.get('start_time')
        length = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[*] Start Time: {start_time}, FPS : {fps}")
    
    
    try:
        logger.info("[*] Connecting to DB...")
        conn = db_conn()

        if input_type == 'video':
            logger.info("[*] Checking Video ID...")
            video_id = video_id_check(conn, video_uid)
            if not video_id:
                logger.error("[*] Video ID not found")
                return
        elif input_type == 'stream':
            logger.info("[*] Checking Camera ID...")
            camera_id = camera_id_check(conn, camera_id, video_path)
            if not camera_id:
                logger.error("[*] Camera ID not found")
                return
    except Exception as e:
        logger.error(e)
        return

    # Video Process
    with tqdm(total=length) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_time = datetime.now()

            if not ret:
                break

            if frame_count % FRAME_CUTOFF == 0:
                crop_data_caption = None
                pil_raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # raw_input = processor(pil_raw_image, return_tensors="pt")
                raw_input = processor(pil_raw_image, return_tensors="pt").to("cuda")
                raw_out = blip_model.generate(**raw_input)

                original_data_caption = (processor.decode(raw_out[0], skip_special_token=True))
                original_data_caption = original_data_caption.replace(' [SEP]', '.')
                logger.info(f"Frame Count: {frame_count}, Original_data_caption: {original_data_caption}")

                # 추출된 Caption Sentence를 바탕으로 Sentimental Model 적용
                predicted_risk = generate_sentimental_score(original_data_caption)

                # Decision RISK
                is_risky = 'N' if predicted_risk[1] >= SENTIMENTAL_THRESHOLD else 'P'
                is_risky_crop = None

                # 추출된 Caption Sentence를 DB에 저장하는 함수 호출
                if input_type == 'video':
                    parse_caption(conn, video_id, frame_count, original_data_caption, predicted_risk[1], is_risky)
                elif input_type == 'stream':
                    stream_parse_caption(conn, camera_id, frame_time, original_data_caption, predicted_risk[1], is_risky)

                # YOLO model detection
                results = yolo_model(frame, verbose=False)
                boxes = results[0].boxes
                for box in boxes:
                    frame_boxes = box.xyxy.cpu().detach().numpy().tolist()
                    frame_class = box.cls.cpu().detach().numpy().tolist()
                    for bbox, case_id in zip(frame_boxes, frame_class):
                        if names[case_id] == 'person':
                            x_vector.append(bbox[0])
                            x_vector.append(bbox[2])
                            y_vector.append(bbox[1])
                            y_vector.append(bbox[3])
                            # print(f'x_vector: {x_vector}, y_vector: {y_vector}')

                # Crop된 이미지가 존재하는 경우
                if len(x_vector) != 0 and len(y_vector) != 0:
                    x_min, x_max = int(min(x_vector)), int(max(x_vector))
                    y_min, y_max = int(min(y_vector)), int(max(y_vector))
                    # print(f'x_min:{x_min}, x_max:{x_max}, y_min:{y_min}, y_max:{y_max}')

                    crop_img = frame[y_min:y_max, x_min:x_max]  # y값 먼저, x값 나중

                    pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    # crop_input = processor(pil_crop_img, return_tensors="pt")
                    crop_input = processor(pil_crop_img, return_tensors="pt").to("cuda")
                    crop_out = blip_model.generate(**crop_input)

                    crop_data_caption = processor.decode(crop_out[0], skip_special_token=True)
                    crop_data_caption = crop_data_caption.replace(' [SEP]', '.')
                    logger.info(f"Frame Count: {frame_count}, Crop_data_caption: {crop_data_caption}")

                    # Crop된 이미지에서 추출된 Caption Sentence를 바탕으로 Sentimental Model 적용
                    predicted_risk_crop = generate_sentimental_score(crop_data_caption)
                            
                    # Decision RISK
                    is_risky_crop = 'N' if predicted_risk_crop[1] >= SENTIMENTAL_THRESHOLD else 'P'

                    # Crop된 이미지에서 추출된 Caption Sentence를 DB에 저장하는 함수 호출
                    if input_type == 'video':
                        parse_caption(conn, video_id, frame_count, crop_data_caption, predicted_risk_crop[1], is_risky_crop)
                    elif input_type == 'stream':
                        stream_parse_caption(conn, camera_id, frame_time, crop_data_caption, predicted_risk_crop[1], is_risky_crop)

            # RISK 구간 판별 -> 각 프레임에서 추출된 문장들 중 하나라도 risk가 존재하면, 해당 프레임은 risk한 것으로 판단함
            # Video input logic
            if input_type == 'video':
                if is_risky == 'N' or is_risky_crop == 'N':
                    start_frame = frame_count
                    start_time = frame_time

                    # frame write here
                    # TODO : file save path 
                    if out is None:
                        filename = f"output_video_{video_uid}_{video_clip_num}.mp4"
                        out = cv2.VideoWriter(filename ,cv2.VideoWriter_fourcc(*'avc1'), fps, (int(cap.get(3)), int(cap.get(4))))
                    out.write(frame)
            
                    if is_N == False:
                        is_N = True
                elif is_risky == 'P' and is_risky_crop == 'P':
                    if is_N == True:
                        # frame write here
                        out.write(frame)

                        if (frame_time - start_time).seconds > CLIP_TIME:
                            risk_section = [start_frame, frame_count]
                            print(f"RISK SECTION: {risk_section}")


                            # risk section video 추출
                            out.release()

                            clip_length = video_length_in_seconds(filename)
                            res = upload_to_s3(VIDEO_BUCKET, filename)
                            if res.get('status'):
                                logger.info(f"[V] Clip URL : {res.get('file_url')}")
                                detect_risky_section(conn, video_uid, video_id, risk_section, res.get('file_url'), clip_length)
                                video_clip_num += 1
                            out = None
                            is_N = False
                            start_frame = 0

            # Camera Stream input logic
            elif input_type == 'stream':
                if is_risky == 'N' or is_risky_crop == 'N':
                    start_time = frame_time

                    # frame write here
                    # TODO : file save path 
                    if out is None:
                        filename = f"output_{start_time.strftime('%y%m%d%H%M%S')}.mp4"
                        # out = cv2.VideoWriter(filename ,cv2.VideoWriter_fourcc(*'mp4v'), fps, (640,480))
                        out = cv2.VideoWriter(filename ,cv2.VideoWriter_fourcc(*'avc1'), fps, (int(cap.get(3)), int(cap.get(4))))
                    out.write(frame)
            
                    if is_N == False:
                        is_N = True
                elif is_risky == 'P' and is_risky_crop == 'P':
                    if is_N == True:
                        # frame write here
                        out.write(frame)

                        if (frame_time - start_time).seconds > CLIP_TIME:
                            risk_section = [start_time, frame_time]
                            print(f"RISK SECTION: {risk_section}")


                            # risk section video 추출
                            out.release()

                            clip_length = video_length_in_seconds(filename)
                            res = upload_to_s3(VIDEO_BUCKET, filename)
                            if res.get('status'):
                                logger.info(f"[V] Clip URL : {res.get('file_url')}")
                                detect_risky_section_stream(conn, camera_id, risk_section, res.get('file_url'), clip_length)
                            out = None
                            is_N = False
                            start_time = 0
                x_vector.clear()
                y_vector.clear()
            pbar.update(1)
            frame_count += 1
    cap.release()


def parse_caption(conn, video_id, frame_num, caption, sentiment_score, is_risky):
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        with conn.cursor() as cursor:
            sql = f"INSERT INTO `{CAPTION_TABLE}` (`video_id`, `frame_number`, `sentence`, `sentiment_score`, `sentiment_result`, `created_at`) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (video_id, frame_num, caption, format(sentiment_score, '.10f'), is_risky, current_time))
            conn.commit()
    except Exception as e:
        logger.error(e)
        return
    
def stream_parse_caption(conn, camera_id, frame_time, caption, sentiment_score, is_risky):
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        logger.info(f"[STREAM] Frame Time : {str(frame_time)}")
        with conn.cursor() as cursor:
            sql = f"INSERT INTO `camera_caption` (`camera_id`, `sentence`, `sentiment_score`, `sentiment_result`, `created_at`) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql, (camera_id, caption, format(sentiment_score, '.10f'), is_risky, current_time))
            conn.commit()
    except Exception as e:
        logger.error(e)
        return

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
    logger.info(f"Input: {caption_sentence}")
    logger.info(f"Predictions: {predictions[0].tolist()}")

    return predictions[0].tolist()


def detect_risky_section(conn, video_uid, video_id, risk_section, clip_url, clip_length):
    try:
        start_frame = risk_section[0]
        end_frame = risk_section[-1]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        with conn.cursor() as cursor:
            sql = f"INSERT INTO `{RISK_TABLE}` (`video_id`, `video_uid`, `clip_url`, `length`, `start_frame`, `end_frame`, `created_at`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (video_id, video_uid, clip_url, clip_length, start_frame, end_frame, current_time))
            conn.commit()
        logger.info(f"[!] Found Risky Section: {start_frame} ~ {end_frame}")
    except Exception as e:
        logger.error(e)
        return
    
def detect_risky_section_stream(conn, camera_id, risk_section, clip_url, clip_length):
    try:
        start_time = str(risk_section[0])
        end_time = str(risk_section[-1])
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        with conn.cursor() as cursor:
            sql = f"INSERT INTO `camera_riskysection` (`camera_id`, `video_uid`, `section_video_url`, `start_time`, `end_time`, `length`, `created_at`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (camera_id, risk_section[0].strftime('%y%m%d%H%M%S'), clip_url, start_time, end_time, clip_length, current_time))
            conn.commit()
        logger.info(f"[!] Found Risky Section: {start_time} ~ {end_time}")
    except Exception as e:
        logger.error(e)
        return