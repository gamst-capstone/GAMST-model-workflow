# 라이브러리 호출
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from dotenv import load_dotenv

import cv2
import pymysql
import os, logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "gamst")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD","")
TABLE_NAME = os.getenv("TABLE_NAME", "video_caption")

processor = None
blip_model = None
yolo_model = None

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


def video_id_check(conn, video_uid):
    with conn.cursor() as cursor:
        sql = f"SELECT id FROM video_video WHERE url LIKE %s"
        cursor.execute(sql, ('%' + video_uid + '%',))
        video_id = cursor.fetchone()[0]
        logger.info(f"[!] Found video_id : {video_id}")
        if not video_id:
            return []
        return video_id

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


# input: VideoPath // output: (5프레임당 Crop된 image Caption 문장 + Original Image Caption 문장)
# if Detection Model에서 Person이 안잡힐 경우 --> Crop된 image Caption 문장: None
def generateCaption(video_path):
    x_vector = []
    y_vector = []
    names = yolo_model.names

    logger.info("[*] Loading Video File...")    
    # video_path = HTTP URL from s3 bucket object
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    video_uid = video_path.split('/')[-1]
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"[*] Video Length: {length}, FPS: {fps}")
    
    try:
        logger.info("[*] Connecting to DB...")
        conn = db_conn()

        video_id = video_id_check(conn, video_uid)
        if not video_id:
            logger.error("[*] Video ID not found")
            return
    except Exception as e:
        logger.error(e)
        return

    with tqdm(total=length) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % 10 == 0:
                crop_data_caption = None
                pil_raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # raw_input = processor(pil_raw_image, return_tensors="pt")
                raw_input = processor(pil_raw_image, return_tensors="pt").to("cuda")
                raw_out = blip_model.generate(**raw_input)

                original_data_caption = (processor.decode(raw_out[0], skip_special_token=True))
                logger.info(f"Frame Count: {frame_count}, Original_data_caption: {original_data_caption}")

                results = yolo_model(frame)
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
                    logger.info(f"Frame Count: {frame_count}, Crop_data_caption: {crop_data_caption}")

                parse_caption(conn, video_id, frame_count, original_data_caption, crop_data_caption)
                x_vector.clear()
                y_vector.clear()
            pbar.update(1)
            frame_count += 1
    cap.release()


def parse_caption(conn, video_id, frame_num, original_data_caption, crop_data_caption):
    if crop_data_caption is None:
        crop_data_caption = ''
    else:
        crop_data_caption = crop_data_caption.replace(' [SEP]', '.')
        original_data_caption = original_data_caption.replace(' [SEP]', '.')
        try:
            with conn.cursor() as cursor:
                sql = f"INSERT INTO `{TABLE_NAME}` (`video_id`, `frame_number`, `original_sentence`, `cropped_sentence`, `created_at`) VALUES (%s, %s, %s, %s, NOW())"
                cursor.execute(sql, (video_id, frame_num, original_data_caption, crop_data_caption))
                conn.commit()
        except Exception as e:
            logger.error(e)
            return