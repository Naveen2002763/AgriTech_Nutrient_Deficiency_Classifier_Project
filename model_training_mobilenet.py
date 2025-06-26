import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Paths
BASE_DIR = "C:/Users/Jabili N/Music/AgriHackathon_NutrientDeficiency/dataset/processed"
MODEL_SAVE_PATH = "C:/Users/Jabili N/Music/AgriHackathon_NutrientDeficiency/models/best_model_mobilenet.h5"
CLASS_INDEX_PATH = "C:/Users/Jabili N/Music/AgriHackathon_NutrientDeficiency/models/class_indices_mobilenet.json"
PLOT_PATH = "C:/Users/Jabili N/Music/AgriHackathon_NutrientDeficiency/models/training_plot_mobilenet.png"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Save class indices
with open(CLASS_INDEX_PATH, 'w') as f:
    json.dump(train_gen.class_indices, f)

# Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # Freeze backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(PLOT_PATH)
print(f"📈 Training plot saved to: {PLOT_PATH}")
plt.close()

# Evaluate
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_pred = np.round(preds).astype(int).flatten()
y_true = val_gen.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["NDVI", "GNDVI"], yticklabels=["NDVI", "GNDVI"], cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(os.path.dirname(PLOT_PATH), "confusion_matrix_mobilenet.png"))
plt.close()

# Classification report
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["NDVI", "GNDVI"]))