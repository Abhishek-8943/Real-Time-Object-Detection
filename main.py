"""
Pascal VOC2012 Object Classification using Transfer Learning
Complete pipeline for training and real-time prediction
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the model"""
    # Dataset paths - YOUR DATASET PATH
    VOC_ROOT = r"D:\L P U\int lpu\CV_Project\archive"
    
    # These will be auto-detected based on your folder structure
    JPEG_IMAGES = None
    ANNOTATIONS = None
    IMAGESETS = None
    
    # Model parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Choose model: 'resnet50', 'vgg16', or 'mobilenetv2'
    BASE_MODEL = 'resnet50'
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Output paths
    MODEL_SAVE_PATH = "voc_classifier_model.h5"
    LABEL_ENCODER_PATH = "label_encoder.pkl"
    HISTORY_PATH = "training_history.pkl"

# Pascal VOC 2012 classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def detect_voc_structure(root_path):
    """
    Auto-detect VOC dataset structure from various formats
    
    Args:
        root_path: Root directory of the dataset
    
    Returns:
        Dictionary with paths to images and annotations
    """
    print(f"Detecting dataset structure in: {root_path}")
    
    paths = {
        'images': None,
        'annotations': None
    }
    
    # Common folder name variations
    image_folders = ['JPEGImages', 'images', 'Images', 'JPEG', 'img', 'VOC2012/JPEGImages']
    annotation_folders = ['Annotations', 'annotations', 'Annotation', 'labels', 'VOC2012/Annotations']
    
    # Search for image folder
    for folder in image_folders:
        test_path = os.path.join(root_path, folder)
        if os.path.exists(test_path):
            # Check if it contains images
            files = os.listdir(test_path)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                paths['images'] = test_path
                print(f"✓ Found images folder: {test_path}")
                break
    
    # Search for annotation folder
    for folder in annotation_folders:
        test_path = os.path.join(root_path, folder)
        if os.path.exists(test_path):
            # Check if it contains XML files
            files = os.listdir(test_path)
            xml_files = [f for f in files if f.lower().endswith('.xml')]
            if xml_files:
                paths['annotations'] = test_path
                print(f"✓ Found annotations folder: {test_path}")
                break
    
    # If not found in common locations, search recursively (up to 2 levels)
    if paths['images'] is None or paths['annotations'] is None:
        print("Searching subdirectories...")
        for root, dirs, files in os.walk(root_path):
            # Limit depth
            depth = root[len(root_path):].count(os.sep)
            if depth > 2:
                continue
            
            # Check for images
            if paths['images'] is None:
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(image_files) > 100:  # Significant number of images
                    paths['images'] = root
                    print(f"✓ Found images in: {root}")
            
            # Check for annotations
            if paths['annotations'] is None:
                xml_files = [f for f in files if f.lower().endswith('.xml')]
                if len(xml_files) > 100:  # Significant number of annotations
                    paths['annotations'] = root
                    print(f"✓ Found annotations in: {root}")
    
    # Validate
    if paths['images'] is None:
        raise FileNotFoundError(
            f"Could not find images folder in {root_path}\n"
            f"Please ensure your dataset contains a folder with .jpg or .png files"
        )
    
    if paths['annotations'] is None:
        raise FileNotFoundError(
            f"Could not find annotations folder in {root_path}\n"
            f"Please ensure your dataset contains a folder with .xml annotation files"
        )
    
    return paths

def parse_voc_annotation(annotation_path):
    """
    Parse Pascal VOC XML annotation file to extract class labels
    
    Args:
        annotation_path: Path to XML annotation file
    
    Returns:
        List of class names present in the image
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    classes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in VOC_CLASSES:
            classes.append(class_name)
    
    return list(set(classes))  # Return unique classes

def load_voc_dataset(config):
    """
    Load Pascal VOC2012 dataset from local folder
    
    Args:
        config: Configuration object with dataset paths
    
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding class labels (primary class)
    """
    print("Loading VOC2012 dataset...")
    
    # Auto-detect dataset structure
    detected_paths = detect_voc_structure(config.VOC_ROOT)
    config.JPEG_IMAGES = detected_paths['images']
    config.ANNOTATIONS = detected_paths['annotations']
    
    image_paths = []
    labels = []
    
    # Get all annotation files
    annotation_files = list(Path(config.ANNOTATIONS).glob("*.xml"))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    for ann_file in annotation_files:
        try:
            # Parse annotation to get classes
            classes = parse_voc_annotation(ann_file)
            
            if not classes:
                continue
            
            # Use the first class as primary label (for single-label classification)
            primary_class = classes[0]
            
            # Get corresponding image path - check multiple extensions
            image_name_base = ann_file.stem
            
            # Try different image extensions
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                image_path = os.path.join(config.JPEG_IMAGES, image_name_base + ext)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    labels.append(primary_class)
                    break
        except Exception as e:
            print(f"Warning: Error processing {ann_file.name}: {e}")
            continue
    
    if len(image_paths) == 0:
        raise ValueError(
            "No valid image-annotation pairs found!\n"
            f"Images folder: {config.JPEG_IMAGES}\n"
            f"Annotations folder: {config.ANNOTATIONS}\n"
            "Please check that image filenames match annotation filenames."
        )
    
    print(f"Loaded {len(image_paths)} images with annotations")
    print(f"\nClass distribution:")
    
    # Show class distribution
    from collections import Counter
    class_counts = Counter(labels)
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    return image_paths, labels

def preprocess_image(image_path, target_size):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image file
        target_size: Tuple of (height, width)
    
    Returns:
        Preprocessed image array
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    return img

def create_data_generators(X_train, y_train, X_val, y_val, config):
    """
    Create data generators with augmentation for training
    
    Args:
        X_train, y_train: Training image paths and labels
        X_val, y_val: Validation image paths and labels
        config: Configuration object
    
    Returns:
        train_generator, val_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load all training images
    print("Loading training images...")
    X_train_images = np.array([
        preprocess_image(path, (config.IMG_HEIGHT, config.IMG_WIDTH)) 
        for path in X_train
    ])
    
    print("Loading validation images...")
    X_val_images = np.array([
        preprocess_image(path, (config.IMG_HEIGHT, config.IMG_WIDTH)) 
        for path in X_val
    ])
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train_images, y_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val_images, y_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    return train_generator, val_generator

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(config, num_classes):
    """
    Build transfer learning model with pre-trained CNN
    
    Args:
        config: Configuration object
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    print(f"Building model with {config.BASE_MODEL}...")
    
    # Load pre-trained base model
    if config.BASE_MODEL == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
        )
    elif config.BASE_MODEL == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
        )
    elif config.BASE_MODEL == 'mobilenetv2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
        )
    else:
        raise ValueError(f"Unknown model: {config.BASE_MODEL}")
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
    )
    
    print(f"Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model, base_model

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, base_model, train_gen, val_gen, config):
    """
    Train the model with callbacks
    
    Args:
        model: Keras model to train
        base_model: Base model for fine-tuning
        train_gen: Training data generator
        val_gen: Validation data generator
        config: Configuration object
    
    Returns:
        Training history
    """
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n" + "="*70)
    print("PHASE 1: Training with frozen base model")
    print("="*70)
    
    # Train with frozen base
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Unfreeze base model for fine-tuning
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning with unfrozen base model")
    print("="*70)
    
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
    )
    
    # Continue training
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    combined_history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'top5_acc': history1.history['top5_acc'] + history2.history['top5_acc'],
        'val_top5_acc': history1.history['val_top5_acc'] + history2.history['val_top5_acc']
    }
    
    return combined_history

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='training_curves.png'):
    """Plot training history curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train Loss')
    axes[0, 1].plot(history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-5 Accuracy
    axes[1, 0].plot(history['top5_acc'], label='Train Top-5 Acc')
    axes[1, 0].plot(history['val_top5_acc'], label='Val Top-5 Acc')
    axes[1, 0].set_title('Top-5 Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning dynamics
    axes[1, 1].plot(np.gradient(history['loss']), label='Train Loss Gradient')
    axes[1, 1].plot(np.gradient(history['val_loss']), label='Val Loss Gradient')
    axes[1, 1].set_title('Loss Gradients (Learning Dynamics)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()

# ============================================================================
# REAL-TIME PREDICTION
# ============================================================================

class VOCClassifier:
    """Wrapper class for making predictions with trained model"""
    
    def __init__(self, model_path, label_encoder_path):
        """Load trained model and label encoder"""
        self.model = keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.classes = self.label_encoder.classes_
        print(f"Model loaded from {model_path}")
        print(f"Classes: {list(self.classes)}")
    
    def predict_image(self, image_path, top_k=5):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            List of (class_name, confidence) tuples
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_array = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = [
            (self.classes[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        return results, img
    
    def predict_webcam(self):
        """Real-time prediction from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit webcam prediction")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            img_resized = cv2.resize(frame, (224, 224))
            img_array = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)[0]
            top_idx = np.argmax(predictions)
            top_class = self.classes[top_idx]
            confidence = predictions[top_idx]
            
            # Display results on frame
            label = f"{top_class}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show top 3 predictions
            top3_indices = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top3_indices):
                text = f"{i+1}. {self.classes[idx]}: {predictions[idx]*100:.1f}%"
                cv2.putText(frame, text, (10, 70 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('VOC Classifier - Webcam', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def predict_and_visualize(self, image_path, save_path=None):
        """Predict and visualize results"""
        results, img = self.predict_image(image_path, top_k=5)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f'Prediction: {results[0][0]} ({results[0][1]*100:.1f}%)')
        
        # Show predictions
        classes = [r[0] for r in results]
        confidences = [r[1] for r in results]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
        bars = ax2.barh(classes, confidences, color=colors)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Top 5 Predictions')
        ax2.set_xlim([0, 1])
        
        # Add percentage labels
        for bar, conf in zip(bars, confidences):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{conf*100:.1f}%', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return results

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Initialize config
    config = Config()
    
    # Check if paths exist
    if not os.path.exists(config.VOC_ROOT):
        print(f"ERROR: VOC dataset not found at {config.VOC_ROOT}")
        print("Please update the VOC_ROOT path in Config class")
        return
    
    # Load dataset
    image_paths, labels = load_voc_dataset(config)
    
    if len(image_paths) == 0:
        print("ERROR: No images loaded. Check your dataset paths.")
        return
    
    # Encode labels
    label_encoder = LabelBinarizer()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Save label encoder
    with open(config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {config.LABEL_ENCODER_PATH}")
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, encoded_labels,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SPLIT/(1-config.TEST_SPLIT),
        stratify=[label_encoder.classes_[np.argmax(y)] for y in y_temp],
        random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    print(f"  Test: {len(X_test)} images")
    
    # Create data generators
    train_gen, val_gen = create_data_generators(
        X_train, y_train, X_val, y_val, config
    )
    
    # Build model
    model, base_model = build_model(config, num_classes)
    model.summary()
    
    # Train model
    history = train_model(model, base_model, train_gen, val_gen, config)
    
    # Save training history
    with open(config.HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nTraining history saved to {config.HISTORY_PATH}")
    
    # Plot training curves
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Load test images
    X_test_images = np.array([
        preprocess_image(path, (config.IMG_HEIGHT, config.IMG_WIDTH)) 
        for path in X_test
    ]) / 255.0
    
    test_loss, test_acc, test_top5 = model.evaluate(X_test_images, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*70}\n")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_predictions():
    """Example: How to use the trained model for predictions"""
    
    # Initialize classifier
    classifier = VOCClassifier(
        model_path="voc_classifier_model.h5",
        label_encoder_path="label_encoder.pkl"
    )
    
    # Example 1: Predict single image
    print("\n=== Example 1: Single Image Prediction ===")
    results, _ = classifier.predict_image("path/to/test/image.jpg")
    print("Top 5 predictions:")
    for i, (class_name, confidence) in enumerate(results, 1):
        print(f"  {i}. {class_name}: {confidence*100:.2f}%")
    
    # Example 2: Predict and visualize
    print("\n=== Example 2: Predict with Visualization ===")
    classifier.predict_and_visualize(
        "path/to/test/image.jpg",
        save_path="prediction_result.png"
    )
    
    # Example 3: Real-time webcam prediction
    print("\n=== Example 3: Webcam Prediction ===")
    print("Uncomment the line below to start webcam prediction")
    # classifier.predict_webcam()

if __name__ == "__main__":
    # Run training
    main()
    
    # After training, you can use the model for predictions
    # Uncomment the line below to run prediction examples
    # example_predictions()