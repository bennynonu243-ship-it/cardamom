import os
import sys
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# 1. Class mapping definition
# (new_id, class_name)
XML_TO_YOLO = {
    14: (0, 'grub'),
    15: (1, 'mole cricket'),
    21: (2, 'red spider'),
    24: (3, 'aphids'),
    54: (4, 'Thrips'),
    64: (5, 'Pseudococcus comstocki Kuwana'),
    71: (6, 'Trialeurodes vaporariorum'),
    76: (7, 'Icerya purchasi Maskell'),
    78: (8, 'Ceroplastes rubens'),
    81: (9, 'Nipaecoccus vastalor'),
    92: (10, 'Scirtothrips dorsalis Hood'),
    101: (11, 'Cicadellidae')
}

RAW_DIR = r'E:\cardamom_pest_train\ip102_raw\Detection\VOC2007'
ANNOT_DIR = os.path.join(RAW_DIR, 'Annotations')
IMG_DIR = os.path.join(RAW_DIR, 'JPEGImages')
OUT_IMG_DIR = r'E:\cardamom_pest_train\filtered_images'
OUT_LABEL_DIR = r'E:\cardamom_pest_train\filtered_labels'

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

t0 = time.time()
xml_files = sorted([f for f in os.listdir(ANNOT_DIR) if f.endswith('.xml')])

kept_images_count = 0
class_image_counts = Counter()
class_box_counts = Counter()
errors = []
samples = []

for fn in xml_files:
    stem = os.path.splitext(fn)[0]
    xml_path = os.path.join(ANNOT_DIR, fn)
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        errors.append((fn, f'XML Parse Error: {e}'))
        continue

    # Image dimensions
    size = root.find('size')
    if size is None:
        errors.append((fn, 'Missing <size> tag'))
        continue
    
    try:
        width = float(size.find('width').text)
        height = float(size.find('height').text)
    except Exception as e:
        errors.append((fn, f'Invalid width/height in XML: {e}'))
        continue

    if width <= 0 or height <= 0:
        errors.append((fn, f'Zero or negative dimension: {width}x{height}'))
        continue

    yolo_lines = []
    classes_in_this_image = set()

    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None or not name_elem.text:
            continue
        c_str = name_elem.text.strip()
        if not c_str.isdigit():
            continue
        orig_id = int(c_str)

        if orig_id in XML_TO_YOLO:
            new_id, class_name = XML_TO_YOLO[orig_id]
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
            except Exception as e:
                errors.append((fn, f'Invalid bndbox coordinates: {e}'))
                continue

            # Clamp coordinates to image boundaries
            xmin = max(0.0, min(width, xmin))
            xmax = max(0.0, min(width, xmax))
            ymin = max(0.0, min(height, ymin))
            ymax = max(0.0, min(height, ymax))

            box_w = xmax - xmin
            box_h = ymax - ymin

            if box_w <= 0 or box_h <= 0:
                continue

            # Convert to YOLO normalized format
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            norm_w = box_w / width
            norm_h = box_h / height

            # Strict bounds clamp
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_lines.append(f'{new_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}')
            classes_in_this_image.add((new_id, class_name))
            class_box_counts[class_name] += 1

    if not yolo_lines:
        # Zero objects from the 12 kept classes -> skip
        continue

    # Verify corresponding image file exists
    src_img_path = os.path.join(IMG_DIR, stem + '.jpg')
    if not os.path.exists(src_img_path):
        # Check case-insensitive
        candidates = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[0].lower() == stem.lower()]
        if candidates:
            src_img_path = os.path.join(IMG_DIR, candidates[0])
        else:
            errors.append((fn, f'Image not found for stem {stem}'))
            continue

    dst_img_path = os.path.join(OUT_IMG_DIR, os.path.basename(src_img_path))
    dst_label_path = os.path.join(OUT_LABEL_DIR, stem + '.txt')

    # Copy image
    shutil.copy2(src_img_path, dst_img_path)

    # Save YOLO label txt
    with open(dst_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines) + '\n')

    kept_images_count += 1
    for _, class_name in classes_in_this_image:
        class_image_counts[class_name] += 1

    if len(samples) < 3:
        samples.append((stem, dst_label_path, yolo_lines))

elapsed = time.time() - t0
print(f'Done processing in {elapsed:.2f} seconds!')
print(f'Total images kept: {kept_images_count}')
print(f'Errors encountered: {len(errors)}')
if errors:
    for err in errors[:10]:
        print('  Error:', err)

print('\n--- Image Count per Kept Class ---')
# In defined order 0-11
for orig_id in [14, 15, 21, 24, 54, 64, 71, 76, 78, 81, 92, 101]:
    new_id, name = XML_TO_YOLO[orig_id]
    imgs = class_image_counts[name]
    boxes = class_box_counts[name]
    print(f'Class {new_id:2d}: {name:<30} | Images: {imgs:5d} | Bounding Boxes: {boxes:5d}')

print('\n--- Sample Label File ---')
if samples:
    s_stem, s_path, s_lines = samples[0]
    print(f'Filename: {s_stem}.txt ({s_path})')
    for l in s_lines:
        print('  ' + l)
