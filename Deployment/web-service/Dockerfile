FROM python:3.12.1-slim-bookworm

# Install pip and dependencies
RUN pip install --upgrade pip

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

# Copy application files
COPY ["predict.py", "ridge_reg.bin", "./"]

EXPOSE 9696

# Run the app with gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]