# 라이브러리 호출
import cv2
from ultralytics import YOLO
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# Captioning Model 로드하는 함수
def loadCaptionModel():
    # BLIP Model Load
    global processor
    processor= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    global blip_model
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


# YOLO Model 로드하는 함수
def loadDetectionModel():
    global yolo_model
    yolo_model= YOLO('yolov8n.pt')


# input: VideoPath // output: (5프레임당 Crop된 image Caption 문장 + Original Image Caption 문장)
    # if Detection Model에서 Person이 안잡힐 경우 --> Crop된 image Caption 문장: None
def generateCaption(video_path):
    x_vector = []
    y_vector = []
    names = yolo_model.names
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        print("파일이 존재하지 않습니다.")
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % 5 == 0:
            crop_data_caption = None
            pil_raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # raw_input = processor(pil_raw_image, return_tensors="pt")
            raw_input = processor(pil_raw_image, return_tensors="pt").to("cuda")
            raw_out = blip_model.generate(**raw_input)

            original_data_caption = (processor.decode(raw_out[0], skip_special_token=True))
            print(f"Frame Count: {frame_count}, Original_data_caption: {original_data_caption}")

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
                print(f"Frame Count: {frame_count}, Crop_data_caption: {crop_data_caption}")

            x_vector.clear()
            y_vector.clear()
        frame_count += 1
    cap.release()