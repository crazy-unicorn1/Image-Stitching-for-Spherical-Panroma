FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim

COPY . ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 python3-opencv -y

RUN pip install -r requirements.txt

RUN pip install fastapi uvicorn

ENV HOST 0.0.0.0

EXPOSE 8080

CMD ["uvicorn", "fastapi_main:app", "--host", "0.0.0.0", "--port", "8080"]