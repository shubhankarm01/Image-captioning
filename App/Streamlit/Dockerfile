FROM python:3.9.4

WORKDIR /image_caption

ADD . ./streamlit

RUN pip install -r ./streamlit/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/image_caption"

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/streamlit_UI.py"]