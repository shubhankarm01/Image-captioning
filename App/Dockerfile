FROM python:3.9.4

WORKDIR /image_caption

ADD . ./App

RUN pip install -r ./App/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/image_caption"

EXPOSE 8001

CMD ["python", "App/api.py"]