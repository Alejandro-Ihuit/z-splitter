FROM python:3.10-slim

WORKDIR /z-splitter
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]

