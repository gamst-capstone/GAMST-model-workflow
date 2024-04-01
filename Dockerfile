FROM omoknooni/gamst-model-base
WORKDIR /app
COPY . /app

RUN pip install flask[async] PyMySQL

CMD ["python", "app.py"]
# CMD ["flask", "--app", "app.py","--debug", "run", "--host", "0.0.0.0"]