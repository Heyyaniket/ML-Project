import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Directory Path
TRAIN_DIR = 'data/train'

# 2. Data Preparation (Added slight zoom/rotation to help the AI learn better)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,      # Slightly rotates images
    zoom_range=0.1,         # Slightly zooms in
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 3. TRANSFER LEARNING: Import MobileNetV2
print("Downloading pre-trained MobileNetV2 brain...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model so we don't destroy its existing knowledge of shapes/textures
base_model.trainable = False

# 4. Build the Custom Top Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(1, activation='sigmoid')
])

# 5. Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. SMART CALLBACKS
# Stop early if validation accuracy doesn't improve for 3 epochs, and keep the best version
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Slow down the learning speed if it gets stuck
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)

print("--------------------------------------------------")
print("Starting PRO Training! Let the Callbacks do the work.")
print("--------------------------------------------------")

# Notice we set epochs=20, but EarlyStopping will likely catch it way before that!
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, lr_reduction]
)

# 7. Save the final Pro model file
model.save('pneumonia_model.h5')
print("Success! Upgraded Pro Model saved as 'pneumonia_model.h5'")