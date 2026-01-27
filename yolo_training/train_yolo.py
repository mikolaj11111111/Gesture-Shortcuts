"""
Trening YOLOv8 na zdjęciach gestów dłoni.

Twoje zdjęcia to wycięte dłonie (bez bounding boxów), więc używamy YOLO w trybie
klasyfikacji obrazów (Image Classification), a nie detekcji obiektów.

Użycie:
    python train_yolo.py
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- ŚCIEŻKI ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "creating_dataset" / "data"
YOLO_DATASET_DIR = Path(__file__).parent / "dataset"

print(f"BASE_DIR: {BASE_DIR.resolve()}")
print(f"DATA_DIR: {DATA_DIR.resolve()}")
print(f"YOLO_DATASET_DIR: {YOLO_DATASET_DIR.resolve()}")

# Kategorie (foldery w data/)
CATEGORIES = ["continue_recording_sign", "stop_recording_sign", "nail_biting_sign", "other"]

# --- KONFIGURACJA TRENINGU ---
MODEL_SIZE = "yolov8n-cls"  # nano - najszybszy, można zmienić na s/m/l/x
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 16
DEVICE = 0  # GPU (cuda:0), użyj "cpu" jeśli brak GPU


def prepare_dataset():
    """Przygotowuje strukturę folderów dla YOLO Classification."""
    print("Przygotowywanie datasetu...")
    
    # Struktura: dataset/train/{klasa}/*.jpg, dataset/val/{klasa}/*.jpg
    train_dir = YOLO_DATASET_DIR / "train"
    val_dir = YOLO_DATASET_DIR / "val"
    
    # Wyczyść jeśli istnieje
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)
    
    for category in CATEGORIES:
        src_images = DATA_DIR / category / "images"
        
        if not src_images.exists():
            print(f"  Brak folderu: {src_images}")
            continue
        
        images = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))
        
        if not images:
            print(f"  Brak zdjęć w: {category}")
            continue
        
        # Podział 80/20 train/val
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Kopiuj do struktury YOLO
        train_cat_dir = train_dir / category
        val_cat_dir = val_dir / category
        train_cat_dir.mkdir(parents=True, exist_ok=True)
        val_cat_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_images:
            shutil.copy(img, train_cat_dir / img.name)
        
        for img in val_images:
            shutil.copy(img, val_cat_dir / img.name)
        
        print(f"  {category}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"Dataset przygotowany w: {YOLO_DATASET_DIR}")
    return YOLO_DATASET_DIR


def train():
    """Trenuje model YOLOv8 Classification."""
    
    # Przygotuj dataset
    dataset_path = prepare_dataset()
    
    # Wczytaj pretrained model
    print(f"\nWczytywanie modelu: {MODEL_SIZE}")
    model = YOLO(f"{MODEL_SIZE}.pt")
    
    # Trening
    print(f"\nRozpoczynam trening ({EPOCHS} epok)...")
    results = model.train(
        data=str(dataset_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(Path(__file__).parent / "runs"),
        name="gesture_classification",
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        patience=10,  # Early stopping
        save=True,
        plots=True,
    )
    
    print("\n" + "="*50)
    print("TRENING ZAKOŃCZONY!")
    print(f"Model zapisany w: runs/gesture_classification/weights/best.pt")
    print("="*50)
    
    return results


if __name__ == "__main__":
    train()
