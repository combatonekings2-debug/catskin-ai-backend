"""
Script Training CNN untuk CatSkin AI
=====================================

Dataset: 5 Kelas Penyakit Kulit Kucing
- Dermatitis (0)
- Flea Allergy (1)
- Ringworm (2)
- Scabies (3)
- Healthy (4)

Arsitektur: CNN dengan Transfer Learning (MobileNetV2)
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================================
# KONFIGURASI
# ============================================================================

CONFIG = {
    # Dataset
    "DATASET_DIR": "../dataset",  # Folder dengan subfolder per kelas
    "IMG_SIZE": (224, 224),
    "BATCH_SIZE": 32,
    "NUM_CLASSES": 5,
    
    # Training
    "EPOCHS": 50,
    "LEARNING_RATE": 0.0001,
    "VALIDATION_SPLIT": 0.2,
    
    # Model
    "MODEL_NAME": "catskin_cnn_model",
    "SAVE_DIR": "../models",
    
    # Class names
    "CLASS_NAMES": ["Dermatitis", "Flea Allergy", "Ringworm", "Scabies", "Healthy"]
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data():
    """
    Prepare data generators untuk training dan validation
    
    Struktur folder yang diharapkan:
    dataset/
    ├── Dermatitis/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── Flea Allergy/
    ├── Ringworm/
    ├── Scabies/
    └── Healthy/
    """
    
    # Data Augmentation untuk training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG["VALIDATION_SPLIT"],
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Hanya rescale untuk validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG["VALIDATION_SPLIT"]
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        CONFIG["DATASET_DIR"],
        target_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        class_mode='categorical',
        subset='training',
        classes=CONFIG["CLASS_NAMES"]
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        CONFIG["DATASET_DIR"],
        target_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        class_mode='categorical',
        subset='validation',
        classes=CONFIG["CLASS_NAMES"]
    )
    
    return train_generator, val_generator

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model():
    """
    Build CNN model dengan Transfer Learning (MobileNetV2)
    
    Arsitektur:
    - Base: MobileNetV2 (pre-trained on ImageNet)
    - Global Average Pooling
    - Dense Layer (512 units, ReLU)
    - Dropout (0.5)
    - Output Layer (5 units, Softmax)
    """
    
    # Load base model (MobileNetV2)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers (optional, bisa di-unfreeze untuk fine-tuning)
    base_model.trainable = False
    
    # Build model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(CONFIG["NUM_CLASSES"], activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def build_simple_cnn():
    """
    Build simple CNN dari scratch (alternatif tanpa transfer learning)
    """
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Block 4
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CONFIG["NUM_CLASSES"], activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_gen, val_gen):
    """Train model dengan callbacks"""
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(CONFIG["SAVE_DIR"], f"{CONFIG['MODEL_NAME']}_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Training
    history = model.fit(
        train_gen,
        epochs=CONFIG["EPOCHS"],
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    return history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, val_gen):
    """Evaluasi model dan tampilkan metrics"""
    
    # Evaluate
    results = model.evaluate(val_gen)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]*100:.2f}%")
    print(f"Precision: {results[2]*100:.2f}%")
    print(f"Recall: {results[3]*100:.2f}%")
    print("="*50)
    
    # Predictions untuk confusion matrix
    val_gen.reset()
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, 
        y_pred_classes, 
        target_names=CONFIG["CLASS_NAMES"]
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CONFIG["CLASS_NAMES"],
        yticklabels=CONFIG["CLASS_NAMES"]
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["SAVE_DIR"], 'confusion_matrix.png'))
    print("\nConfusion matrix saved!")

def plot_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["SAVE_DIR"], 'training_history.png'))
    print("Training history plot saved!")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("🐱 CatSkin AI - CNN Training")
    print("="*60)
    print(f"\nDataset: {CONFIG['DATASET_DIR']}")
    print(f"Classes: {CONFIG['CLASS_NAMES']}")
    print(f"Image Size: {CONFIG['IMG_SIZE']}")
    print(f"Epochs: {CONFIG['EPOCHS']}")
    print(f"Batch Size: {CONFIG['BATCH_SIZE']}")
    print("="*60)
    
    # Create save directory
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    
    # Prepare data
    print("\n📂 Preparing data...")
    train_gen, val_gen = prepare_data()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Build model
    print("\n🏗️ Building model...")
    model = build_model()
    # model = build_simple_cnn()  # Alternatif
    
    model.summary()
    
    # Train
    print("\n🚀 Starting training...")
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    evaluate_model(model, val_gen)
    
    # Plot history
    plot_history(history)
    
    # Save final model
    final_model_path = os.path.join(CONFIG["SAVE_DIR"], "cnn_model.h5")
    model.save(final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")
    
    # Save as SavedModel format (untuk TensorFlow Serving)
    savedmodel_path = os.path.join(CONFIG["SAVE_DIR"], "saved_model")
    model.save(savedmodel_path)
    print(f"💾 SavedModel saved to: {savedmodel_path}")
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print("="*60)

if __name__ == "__main__":
    main()
