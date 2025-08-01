FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# 👉 Définir PYTHONPATH pour que le dossier /app soit accessible dans les imports
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
