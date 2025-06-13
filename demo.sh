#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate

# Start API (background)
echo "Starting FastAPI server..."
nohup uvicorn src.api:app --host 0.0.0.0 --port 8000 > logs/api_stdout.log 2> logs/api_stderr.log &

# Start Prometheus (background)
echo "Starting Prometheus..."
nohup ./prometheus-2.45.0.linux-amd64/prometheus \
  --config.file=monitoring/prometheus.yml \
  --web.listen-address=":9090" > logs/prometheus.log 2>&1 &

# Start Alertmanager (background)
echo "Starting Alertmanager..."
nohup ./alertmanager-0.27.0.linux-amd64/alertmanager \
  --config.file=monitoring/alertmanager.yml \
  --web.listen-address=":9093" > logs/alertmanager.log 2>&1 &

# Start Node Exporter (background)
echo "Starting Node Exporter..."
nohup ./node_exporter-1.6.1.linux-amd64/node_exporter > logs/node_exporter.log 2>&1 &

# Start Grafana (background)
echo "Starting Grafana..."
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

echo "All services started!"
echo "API:           http://localhost:8000"
echo "Prometheus:    http://localhost:9090"
echo "Alertmanager:  http://localhost:9093"
echo "Grafana:       http://localhost:3000 (admin/admin)"

# Wait for services to start
sleep 5

# Test API
echo "Testing API..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 1000,
    "bedrooms": 2,
    "bathrooms": 1,
    "stories": 1,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 1,
    "prefarea": "yes",
    "furnishingstatus": "furnished"
  }'

echo -e "\n\nTo stop all services, run:"
echo "pkill -f 'uvicorn|prometheus|alertmanager|node_exporter'"
echo "sudo systemctl stop grafana-server" 