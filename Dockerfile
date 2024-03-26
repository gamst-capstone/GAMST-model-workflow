FROM python:latest
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6-y

RUN pip install -r requirements.txt

CMD ["python", "app.py"]