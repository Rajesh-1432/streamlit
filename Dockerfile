FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9050

CMD ["streamlit", "run", "app.py", "--server.port=9050", "--server.address=0.0.0.0"]