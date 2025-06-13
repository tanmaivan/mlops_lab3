import requests
import time
import random
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_valid_request():
    """Generate a valid request payload"""
    return {
        "area": random.randint(1000, 10000),
        "bedrooms": random.randint(1, 6),
        "bathrooms": random.randint(1, 4),
        "stories": random.randint(1, 4),
        "mainroad": random.choice(["yes", "no"]),
        "guestroom": random.choice(["yes", "no"]),
        "basement": random.choice(["yes", "no"]),
        "hotwaterheating": random.choice(["yes", "no"]),
        "airconditioning": random.choice(["yes", "no"]),
        "parking": random.randint(0, 3),
        "prefarea": random.choice(["yes", "no"]),
        "furnishingstatus": random.choice(["furnished", "semi-furnished", "unfurnished"])
    }

def generate_invalid_request(request_type="all_invalid"):
    """Generate different types of invalid request payloads"""
    if request_type == "all_invalid":
        return {
            "area": "invalid",
            "bedrooms": "invalid",
            "bathrooms": "invalid",
            "stories": "invalid",
            "mainroad": "invalid",
            "guestroom": "invalid",
            "basement": "invalid",
            "hotwaterheating": "invalid",
            "airconditioning": "invalid",
            "parking": "invalid",
            "prefarea": "invalid",
            "furnishingstatus": "invalid"
        }
    elif request_type == "missing_fields":
        return {
            "area": random.randint(1000, 10000),
            "bedrooms": random.randint(1, 6)
            # Missing other required fields
        }
    elif request_type == "invalid_values":
        return {
            "area": -1000,  # Negative area
            "bedrooms": 100,  # Unrealistic number of bedrooms
            "bathrooms": -1,  # Negative bathrooms
            "stories": 0,  # Invalid stories
            "mainroad": "maybe",  # Invalid value
            "guestroom": "sometimes",  # Invalid value
            "basement": "unknown",  # Invalid value
            "hotwaterheating": "yes",  # Valid
            "airconditioning": "no",  # Valid
            "parking": 100,  # Unrealistic parking
            "prefarea": "yes",  # Valid
            "furnishingstatus": "partially"  # Invalid value
        }
    elif request_type == "malformed_json":
        return "This is not a JSON object"
    else:
        return {}

def make_request(request_type="valid", timeout=5):
    """Make a single request to the API"""
    url = "http://localhost:8000/predict"
    
    if request_type == "valid":
        payload = generate_valid_request()
    else:
        payload = generate_invalid_request(request_type)
    
    start_time = time.time()
    try:
        if request_type == "malformed_json":
            response = requests.post(url, data=payload, timeout=timeout)
        else:
            response = requests.post(url, json=payload, timeout=timeout)
            
        latency = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"Success - Latency: {latency:.3f}s - Response: {response.json()}")
            return {
                "status": "success",
                "latency": latency,
                "response": response.json()
            }
        else:
            error_type = "validation_error"
            if response.status_code == 422:
                error_type = "validation_error"
            elif response.status_code == 500:
                error_type = "server_error"
            elif response.status_code == 503:
                error_type = "service_unavailable"
            else:
                error_type = "unknown_error"
                
            logger.error(f"Error {response.status_code} ({error_type}) - Latency: {latency:.3f}s - Response: {response.text}")
            return {
                "status": "error",
                "error_type": error_type,
                "latency": latency,
                "response": response.text
            }
    except requests.exceptions.Timeout:
        logger.error(f"Timeout after {timeout}s")
        return {
            "status": "timeout",
            "error_type": "timeout_error",
            "latency": timeout,
            "response": None
        }
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error: Failed to connect to {url}")
        return {
            "status": "error",
            "error_type": "connection_error",
            "latency": time.time() - start_time,
            "response": "Connection refused"
        }
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "latency": time.time() - start_time,
            "response": str(e)
        }

def run_load_test(num_requests=100, concurrent_requests=10, error_rate=0.4, timeout=5):
    """Run the load test with specified parameters"""
    logger.info(f"Starting load test with {num_requests} requests, {concurrent_requests} concurrent requests, {error_rate*100}% error rate")
    
    results = []
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for _ in range(num_requests):
            if random.random() < error_rate:
                # Randomly choose an error type
                error_type = random.choice(["all_invalid", "missing_fields", "invalid_values", "malformed_json"])
                futures.append(executor.submit(make_request, error_type, timeout))
            else:
                futures.append(executor.submit(make_request, "valid", timeout))
            
            # Add small delay between requests to create load
            time.sleep(random.uniform(0.1, 0.3))
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Calculate statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] in ["error", "timeout"])
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    
    # Count different types of errors
    error_types = {}
    for r in results:
        if r["status"] == "error":
            error_type = r.get("error_type", "unknown_error")
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    latencies = [r["latency"] for r in results if r["status"] == "success"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    stats = {
        "total_requests": num_requests,
        "success_count": success_count,
        "error_count": error_count,
        "timeout_count": timeout_count,
        "error_types": error_types,
        "success_rate": success_count / num_requests,
        "error_rate": error_count / num_requests,
        "avg_latency": avg_latency,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0
    }
    
    # Save results to file
    with open('load_test_results.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Load test completed")
    logger.info(f"Results: {json.dumps(stats, indent=2)}")
    
    return stats

if __name__ == "__main__":
    run_load_test(num_requests=100, concurrent_requests=10, error_rate=0.4, timeout=5) 