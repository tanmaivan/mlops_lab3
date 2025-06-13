from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import RequestValidationError
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os
from typing import List, Optional
import time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import logging
import sys
from logging.handlers import SysLogHandler
from pythonjsonlogger import jsonlogger
from fluent.sender import FluentSender
import psutil
from fluent import sender
from starlette.requests import ClientDisconnect
import traceback

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        timestamp=True
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s - Latency: %(latency).3fs - Response: %(response)s'
    )
    error_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s\nError: %(error)s\nTraceback:\n%(traceback)s'
    )
    
    # Create handlers
    # 1. Syslog handler
    syslog_handler = SysLogHandler(address='/dev/log')
    syslog_handler.setFormatter(console_formatter)
    syslog_handler.setLevel(logging.WARNING)
    
    # 2. Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # 3. Error file handler (stderr)
    error_file_handler = logging.FileHandler('logs/api_stderr.log')
    error_file_handler.setFormatter(error_formatter)
    error_file_handler.setLevel(logging.ERROR)
    
    # 4. Info file handler
    info_file_handler = logging.FileHandler('logs/api_info.log')
    info_file_handler.setFormatter(json_formatter)
    info_file_handler.setLevel(logging.INFO)
    
    # 5. Stdout file handler (log stream)
    stdout_file_handler = logging.FileHandler('logs/api_stdout.log')
    stdout_file_handler.setFormatter(console_formatter)
    stdout_file_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(syslog_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(info_file_handler)
    root_logger.addHandler(stdout_file_handler)
    
    # Configure application logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    return logger

# Initialize logging
logger = setup_logging()

# Configure Fluentd logger
fluent_logger = sender.FluentSender('housing-api', host='localhost', port=24224)

app = FastAPI(title="Housing Price Prediction API")

# Prometheus metrics
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)

INFERENCE_TIME = Histogram(
    'inference_time_seconds',
    'Model inference time in seconds'
)

PREDICTION_COUNTER = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)

ERROR_COUNTER = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

VALIDATION_ERROR_COUNTER = Counter(
    'validation_errors_total',
    'Total number of validation errors'
)

REQUEST_VALIDATION_ERROR_COUNTER = Counter(
    'request_validation_errors_total',
    'Total number of request validation errors'
)

SERVER_ERROR_COUNTER = Counter(
    'server_errors_total',
    'Total number of server errors'
)

CONNECTION_ERROR_COUNTER = Counter(
    'connection_errors_total',
    'Total number of connection errors'
)

CONFIDENCE_SCORE = Gauge(
    'prediction_confidence',
    'Confidence score of the prediction'
)

# System metrics
CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'memory_usage_percent',
    'Memory usage percentage'
)

DISK_USAGE = Gauge(
    'disk_usage_percent',
    'Disk usage percentage'
)

# Load the model and preprocessors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = os.path.join(BASE_DIR, "models", "gradient_boosting_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load scalers
scaler_X_path = os.path.join(BASE_DIR, "models", "scaler_X.pkl")
scaler_y_path = os.path.join(BASE_DIR, "models", "scaler_y.pkl")

with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)

# Load encoders
encoders = {}
categorical_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea',
    'furnishingstatus'
]

for col in categorical_cols:
    encoder_path = os.path.join(BASE_DIR, "models", f"encoder_{col}.pkl")
    with open(encoder_path, 'rb') as f:
        encoders[col] = pickle.load(f)

class HousingInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

class HousingResponse(BaseModel):
    predicted_price: float

def update_system_metrics():
    """Update system metrics"""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)

async def safe_read_request_body(request: Request) -> str:
    """Safely read request body with error handling"""
    try:
        body = await request.body()
        return body.decode() if body else "Empty body"
    except ClientDisconnect:
        return "Client disconnected before body could be read"
    except Exception as e:
        return f"Error reading body: {str(e)}"

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    ERROR_COUNTER.inc()
    VALIDATION_ERROR_COUNTER.inc()
    body = await safe_read_request_body(request)
    
    logger.error("Validation error", extra={
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "input": body,
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Send error log to Fluentd
    fluent_logger.emit("validation_error", {
        "error": str(exc),
        "input": body,
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    ERROR_COUNTER.inc()
    REQUEST_VALIDATION_ERROR_COUNTER.inc()
    body = await safe_read_request_body(request)
    
    logger.error("Request validation error", extra={
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "body": body,
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Send error log to Fluentd
    fluent_logger.emit("request_validation_error", {
        "error": str(exc),
        "body": body,
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.post("/predict", response_model=HousingResponse)
async def predict_price(input_data: HousingInput):
    start_time = time.time()
    try:
        # Update system metrics
        update_system_metrics()
        
        # Convert input to DataFrame
        input_dict = input_data.dict()
        df_input = pd.DataFrame([input_dict])
        
        # Encode categorical variables using saved encoders
        for col in categorical_cols:
            df_input[col] = encoders[col].transform(df_input[col])
        
        # Scale features
        X_scaled = scaler_X.transform(df_input)
        
        # Measure inference time
        inference_start = time.time()
        y_pred_scaled = model.predict(X_scaled)
        inference_time = time.time() - inference_start
        INFERENCE_TIME.observe(inference_time)
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # Calculate confidence score (using model's predict_proba if available)
        try:
            confidence = model.predict_proba(X_scaled).max()
            CONFIDENCE_SCORE.set(confidence)
        except:
            CONFIDENCE_SCORE.set(1.0)  # Default confidence if predict_proba not available
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        # Log successful prediction
        success_details = {
            "input": input_dict,
            "prediction": float(y_pred[0][0]),
            "latency": time.time() - start_time,
            "inference_time": inference_time,
            "confidence": float(confidence) if 'confidence' in locals() else 1.0,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("Success", extra={
            "latency": time.time() - start_time,
            "response": {"predicted_price": float(y_pred[0][0])}
        })
        
        # Send log to Fluentd
        fluent_logger.emit("prediction", success_details)
        
        return {"predicted_price": float(y_pred[0][0])}
    
    except Exception as e:
        ERROR_COUNTER.inc()
        SERVER_ERROR_COUNTER.inc()
        logger.error("Prediction failed", extra={
            "error": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            "input": input_dict if 'input_dict' in locals() else None,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Send error log to Fluentd
        fluent_logger.emit("prediction_error", {
            "error": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            "input": input_dict if 'input_dict' in locals() else None,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to Housing Price Prediction API"}

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 