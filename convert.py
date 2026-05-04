import tensorflow as tf

# 1. Đường dẫn file model H5 của bạn
h5_model_path = 'model_plant_keras2.h5' 

# 2. Load model
model = tf.keras.models.load_model(h5_model_path, compile=False)

# 3. Cấu hình Converter để tương thích với bản cũ (Legacy)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# QUAN TRỌNG: Ép buộc sử dụng các Ops cũ hơn
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS # Chỉ dùng built-ins chuẩn
]

# Thêm dòng này để ngăn lỗi Version 12
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

# 4. Lưu lại file mới (Ghi đè lên file cũ)
with open('model_plant.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Đã tạo lại file model_plant.tflite với phiên bản Op tương thích thấp hơn!")