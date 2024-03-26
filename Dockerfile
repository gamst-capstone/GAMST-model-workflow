FROM omoknooni/gamst-model-base
WORKDIR /app
COPY . /app

RUN pip install flask[async]

CMD ["python", "app.py"]
