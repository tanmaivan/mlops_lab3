# MLOps Lab 02 - Monitoring and Logging
demo: https://youtu.be/G5dKctv_khQ
## 1. Cài đặt môi trường

### 1.1. Yêu cầu hệ thống
- Python 3.12
- Linux (Ubuntu 22.04 LTS)
- Docker (tùy chọn)

### 1.2. Cài đặt các công cụ monitoring

#### Prometheus
```bash
# Tải Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
```

#### Alertmanager
```bash
# Tải Alertmanager
wget https://github.com/prometheus/alertmanager/releases/download/v0.27.0/alertmanager-0.27.0.linux-amd64.tar.gz
tar xvfz alertmanager-0.27.0.linux-amd64.tar.gz
```

#### Node Exporter
```bash
# Tải Node Exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz
```

#### Grafana
```bash
# Cài đặt Grafana
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana
```

### 1.3. Cài đặt môi trường Python

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate

# Cài đặt các thư viện với phiên bản cụ thể
pip install -r requirements.txt
```

Các thư viện chính được sử dụng:
- fastapi==0.109.2
- uvicorn==0.27.1
- scikit-learn==1.3.2
- pandas==2.2.0
- numpy==1.26.3
- prometheus-client==0.19.0
- python-fluent-logger==0.9.6
- psutil==5.9.8

## 2. Khởi chạy hệ thống

### 2.1. Khởi động nhanh
Sử dụng script demo.sh để khởi động tất cả các service:
```bash
chmod +x demo.sh
./demo.sh
```

Script sẽ khởi động:
- FastAPI server (port 8000)
- Prometheus (port 9090)
- Alertmanager (port 9093)
- Node Exporter
- Grafana (port 3000)

### 2.2. Khởi động thủ công
Nếu muốn khởi động từng service riêng lẻ:

```bash
# Khởi động API
source .venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Khởi động Prometheus
./prometheus-2.45.0.linux-amd64/prometheus --config.file=monitoring/prometheus.yml

# Khởi động Alertmanager
./alertmanager-0.27.0.linux-amd64/alertmanager --config.file=monitoring/alertmanager.yml

# Khởi động Node Exporter
./node_exporter-1.6.1.linux-amd64/node_exporter

# Khởi động Grafana
sudo systemctl start grafana-server
```

## 3. Demo Video

Video demo bao gồm các phần sau:

### 3.1. Dashboard Monitoring
- Hiển thị các metrics chính:
  - Request rate
  - Error rate
  - Latency
  - CPU/Memory usage
  - System metrics

### 3.2. Load Testing
- Chạy script load_test.py để giả lập traffic
- Quan sát sự thay đổi của các metrics trên dashboard
- Phân tích performance dưới tải

### 3.3. Logging
- Capture logs từ API (stdout/stderr)
- System logs
- Error logs khi API gặp lỗi

### 3.4. Error Handling
- Giả lập các trường hợp lỗi
- Quan sát error rate trên dashboard
- Kiểm tra error logs

## 4. Dừng hệ thống

```bash
# Dừng tất cả các service
pkill -f 'uvicorn|prometheus|alertmanager|node_exporter'
sudo systemctl stop grafana-server
```

## 5. Cấu trúc thư mục

```
.
├── data/               # Dữ liệu training
├── logs/              # Log files
├── monitoring/        # Cấu hình Prometheus và Alertmanager
├── models/           # Model đã train
├── src/              # Source code
│   ├── api.py        # FastAPI application
│   └── model.py      # Model loading và prediction
├── demo.sh           # Script khởi động nhanh
├── load_test.py      # Script load testing
└── requirements.txt  # Python dependencies
```

## 6. Troubleshooting

### 6.1. Port conflicts
Nếu gặp lỗi "Address already in use":
```bash
# Tìm process đang sử dụng port
sudo lsof -i :8000
sudo lsof -i :9090
sudo lsof -i :9093
sudo lsof -i :3000

# Kill process
sudo kill -9 <PID>
```

### 6.2. Log files
- API logs: `logs/api_stdout.log` và `logs/api_stderr.log`
- Prometheus logs: `logs/prometheus.log`
- Alertmanager logs: `logs/alertmanager.log`
- Node Exporter logs: `logs/node_exporter.log`
- Grafana logs: `/var/log/grafana/grafana.log`

## 7. Grafana Dashboard

### 7.1. Export Dashboard
1. Đăng nhập vào Grafana (http://localhost:3000, admin/admin)
2. Vào dashboard cần export
3. Click vào biểu tượng Share (góc trên bên phải)
4. Chọn tab "Export"
5. Click "Save to file" để tải file JSON
6. Lưu file vào thư mục `monitoring/dashboards/`

### 7.2. Import Dashboard
1. Đăng nhập vào Grafana
2. Click "+" ở sidebar, chọn "Import"
3. Click "Upload JSON file"
4. Chọn file dashboard JSON từ thư mục `monitoring/dashboards/`
5. Chọn data source (Prometheus)
6. Click "Import"

### 7.3. Dashboard Files
Các file dashboard được lưu trong thư mục `monitoring/dashboards/`:
- `api_metrics.json`: Dashboard cho API metrics (request rate, error rate, latency)
- `system_metrics.json`: Dashboard cho system metrics (CPU, memory, disk)
- `model_metrics.json`: Dashboard cho model metrics.

### 7.4. Backup Dashboard
Để backup toàn bộ cấu hình Grafana:
```bash
# Backup cấu hình Grafana
sudo systemctl stop grafana-server
sudo tar -czf grafana_backup.tar.gz /etc/grafana /var/lib/grafana
sudo systemctl start grafana-server

# Restore cấu hình
sudo systemctl stop grafana-server
sudo tar -xzf grafana_backup.tar.gz -C /
sudo systemctl start grafana-server
```
