import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
print(f"Metal加速状态: {tf.config.list_physical_devices('GPU')}")
