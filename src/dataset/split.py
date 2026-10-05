import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config(config_path="d:/MV/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    dataset_cfg = config['dataset']
    
    orig_dir = Path(dataset_cfg['original_dir'])
    classes = dataset_cfg['classes']
    ratios = dataset_cfg['split_ratios']
    seed = dataset_cfg['seed']
    out_dir = Path(dataset_cfg['output_dir'])
    
    train_ratio = ratios['train']
    val_ratio = ratios['val']
    test_ratio = ratios['test']
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
    
    # Collect all image paths
    # The original dataset has train and test folders. 
    # We will combine all images from both seg_train and seg_test for the selected classes,
    # and then do our own clean split.
    
    all_images = []
    labels = []
    
    for split_dir in ['seg_train/seg_train', 'seg_test/seg_test']:
        for cls in classes:
            class_dir = orig_dir / split_dir / cls
            if not class_dir.exists():
                logging.warning(f"Directory not found: {class_dir}")
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                all_images.append(img_path)
                labels.append(cls)
                
    logging.info(f"Total images collected: {len(all_images)}")
    
    if len(all_images) == 0:
        logging.error("No images found. Please check the dataset path.")
        return

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_images, val_test_images, train_labels, val_test_labels = train_test_split(
        all_images, labels, test_size=val_test_ratio, stratify=labels, random_state=seed
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_images, test_images, val_labels, test_labels = train_test_split(
        val_test_images, val_test_labels, test_size=relative_test_ratio, stratify=val_test_labels, random_state=seed
    )
    
    splits = {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels)
    }
    
    # Verify no data leakage (intersection between sets)
    train_set = set(train_images)
    val_set = set(val_images)
    test_set = set(test_images)
    
    assert len(train_set.intersection(val_set)) == 0, "Leakage between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage between val and test!"
    logging.info("Data leakage verification passed. No overlapping images across splits.")
    
    # Create directories and copy files
    if out_dir.exists():
        logging.info("Removing existing output directory...")
        shutil.rmtree(out_dir)
        
    for split_name, (imgs, lbls) in splits.items():
        logging.info(f"Processing {split_name} split...")
        class_counts = {cls: 0 for cls in classes}
        
        for img_path, cls in zip(imgs, lbls):
            target_dir = out_dir / split_name / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / img_path.name
            # Ensure unique filename if duplicate exists
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1
                
            shutil.copy2(img_path, target_path)
            class_counts[cls] += 1
            
        logging.info(f"Split '{split_name}' class counts: {class_counts}")

if __name__ == "__main__":
    main()
