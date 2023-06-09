FROM pypy:latest
WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python", "semantic.py"]