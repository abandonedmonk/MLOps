# MLOps Zoomcamp Homework 4: Model Deployment

This project implements a pipeline for predicting ride durations using the NYC Yellow Taxi Trip Records dataset. It includes data loading, transformation, model training, registration with MLflow, batch predictions, and hosting a web service locally using Flask. Below are the instructions to run the pipeline locally.

## Prerequisites

- Python 3.10
- Docker
- `pipenv` for virtual environment management
- Internet access to download datasets

## Setup

1. **Install `pipenv`**:

   ```bash
   pip install pipenv
   ```

2. **Create virtual environment and install dependencies**:

   ```bash
   pipenv install pandas pyarrow scikit-learn==1.5.0 mlflow flask
   ```

3. **Activate virtual environment**:

   ```bash
   pipenv shell
   ```

4. **Start MLflow server** (in a separate terminal):
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
   ```

## Project Structure

```
mlops-homework/
├── load_data.py
├── transform_data.py
├── train_model.py
├── register_model.py
├── predict.py
├── app.py
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── model.bin (generated)
├── predictions_2023_03.parquet (generated)
├── predictions_2023_04.parquet (generated)
├── predictions_2023_05.parquet (generated)
```

## Running the Pipeline

### 1. Train and Register Model

Train a model on March 2023 data and register it with MLflow.

```bash
python train_model.py --year 2023 --month 3 --model-path model.bin
python register_model.py --year 2023 --month 3
```

### 2. Batch Predictions

Run batch predictions for homework questions.

- **March 2023 (Q1: Standard deviation, Q2: File size)**:

  ```bash
  python predict.py --year 2023 --month 3 --experiment-name yellow-taxi-linear-regression --output-path predictions_2023_03.parquet
  ```

  - Check file size for Q2:
    ```bash
    ls -lh predictions_2023_03.parquet
    ```

- **April 2023 (Q5: Mean predicted duration)**:

  ```bash
  python predict.py --year 2023 --month 4 --experiment-name yellow-taxi-linear-regression --output-path predictions_2023_04.parquet
  ```

- **May 2023 (Q6: Mean predicted duration, Docker)**:
  - Build Docker image:
    ```bash
    docker build -t mlops-homework .
    ```
  - Run container:
    ```bash
    docker run --rm mlops-homework --year 2023 --month 5 --model-path /app/model.bin --output-path predictions_2023_05.parquet
    ```

### 3. Host Web Service

Run the Flask app to serve predictions locally.

```bash
python app.py
```

Test the endpoint:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"PULocationID": "161", "DOLocationID": "236", "trip_distance": 2.5}'
```

### 4. Convert Notebook to Script (Q3)

Convert `starter.ipynb` to a script (if applicable):

```bash
jupyter nbconvert --to script starter.ipynb
```

## Homework Answers

- **Q1**: Standard deviation for March 2023 ≈ 6.24
- **Q2**: Size of `predictions_2023_03.parquet` ≈ 66M
- **Q3**: `jupyter nbconvert --to script starter.ipynb`
- **Q4**: First hash for `scikit-learn==1.5.0` in `Pipfile.lock`:
  ```
  sha256:057b991ac64b3e75c9c04b5f9395eaf19a6179244f9cd494e748d2773d8065d0
  ```
- **Q5**: Mean predicted duration for April 2023 ≈ 14.29
- **Q6**: Mean predicted duration for May 2023 ≈ 14.24

## Troubleshooting

- **MLflow**: Ensure server is running at `http://127.0.0.1:5000`.
- **Docker**: Verify Docker is installed and base image is accessible.
- **Data**: If URLs fail, download parquet files from `https://d37ci6vzurychx.cloudfront.net/trip-data/`.
- **Dependencies**: Use `pipenv install` to resolve version conflicts.
