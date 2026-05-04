import os
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

#app = FastAPI()
app=FastAPI()

# 1. Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_plant.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.json")

# Khởi tạo các biến toàn cục
interpreter = None
input_details = None
output_details = None
labels = {}

@app.on_event("startup")
async def startup_event():
    global interpreter, input_details, output_details, labels
    
    # Load labels với xử lý lỗi
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"✅ Loaded {len(labels)} labels.")
        else:
            print("⚠️ Warning: labels.json not found!")
    except Exception as e:
        print(f"❌ Error loading labels: {e}")

    # Load TFLite model với xử lý lỗi
    try:
        if os.path.exists(MODEL_PATH):
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("✅ TFLite Model loaded successfully!")
        else:
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        return {"error": "Model not loaded on server."}

    try:
        # 1. Đọc và xử lý ảnh
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
        
        # 2. Tiền xử lý theo chuẩn MobileNetV2: [-1, 1]
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Chạy inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # 4. Lấy kết quả
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_id = np.argmax(predictions)
        score = float(predictions[class_id]) # Lấy độ tự tin của class cao nhất

        # 5. Ngưỡng tin cậy (Confidence Threshold)
        # Nếu thấp hơn 80%, có thể ảnh không phải lá cây hoặc bệnh lạ
        if score < 0.80:
            return {
                "label": "Không xác định", 
                "confidence": round(score * 100, 2),
                "message": "Ảnh không rõ ràng hoặc chưa có trong dữ liệu."
            }

        result_label = labels.get(str(class_id), f"Unknown Class {class_id}")
        
        return {
            "label": result_label, 
            "confidence": round(score * 100, 2)
        }

    except Exception as e:
        return {"error": f"Inference error: {str(e)}"}
    finally:
        await file.close() # Giải phóng bộ nhớ file

@app.get("/")
def home():
    status = "Ready" if interpreter else "Model Missing"
    return {
        "status": "TFLite Server Online",
        "model_status": status,
        "labels_count": len(labels)
    }