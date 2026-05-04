import tensorflow as tf

# Tên file model .h5 của bạn (đảm bảo file này đã nằm trong thư mục C:\modelTFlite)
h5_model_path = 'model_plant_keras2.h5' 
# Load model
model = tf.keras.models.load_model(h5_model_path, compile=False)

# Convert sang TFLite với cấu hình tương thích ngược
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Lưu thành file mới
with open('model_plant_final.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Đã tạo xong model_plant_final.tflite!")