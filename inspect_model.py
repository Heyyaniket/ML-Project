import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf

# Load your model
print("Loading model...")
model = tf.keras.models.load_model('pneumonia_model.h5')

# Print the internal architecture
print("\n--- MODEL ARCHITECTURE ---")
model.summary()