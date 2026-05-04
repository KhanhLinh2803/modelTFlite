# 1. Sử dụng Python bản slim để tiết kiệm dung lượng (chỉ khoảng 150MB)
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Cài đặt các thư viện hệ thống cần thiết (rất ít)
# tflite-runtime chỉ cần một vài thư viện cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ mã nguồn và thư mục model vào container
COPY . .

# 6. Mở cổng 8000 cho FastAPI
EXPOSE 8000

# 7. Lệnh chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]