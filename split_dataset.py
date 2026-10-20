import os
import shutil
import random

src_img = r'E:/cardamom_pest_train/filtered_images'
src_lbl = r'E:/cardamom_pest_train/filtered_labels'
dataset_dir = r'E:/cardamom_pest_train/dataset'

splits = {
    'train': (os.path.join(dataset_dir, 'train', 'images'), os.path.join(dataset_dir, 'train', 'labels')),
    'val': (os.path.join(dataset_dir, 'val', 'images'), os.path.join(dataset_dir, 'val', 'labels')),
    'test': (os.path.join(dataset_dir, 'test', 'images'), os.path.join(dataset_dir, 'test', 'labels'))
}

for img_path, lbl_path in splits.values():
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(lbl_path, exist_ok=True)

stems = sorted([os.path.splitext(f)[0] for f in os.listdir(src_img) if f.endswith('.jpg')])
random.seed(42)
shuffled = list(stems)
random.shuffle(shuffled)

n_total = len(shuffled)
n_val = round(n_total * 0.1)    # 581
n_test = round(n_total * 0.1)   # 581
n_train = n_total - n_val - n_test # 4645

split_data = {
    'train': shuffled[:n_train],
    'val': shuffled[n_train:n_train+n_val],
    'test': shuffled[n_train+n_val:]
}

print('Organizing files into splits...')
for split_name, stem_list in split_data.items():
    dst_img_dir, dst_lbl_dir = splits[split_name]
    for s in stem_list:
        shutil.copy2(os.path.join(src_img, s + '.jpg'), os.path.join(dst_img_dir, s + '.jpg'))
        shutil.copy2(os.path.join(src_lbl, s + '.txt'), os.path.join(dst_lbl_dir, s + '.txt'))
    print(f'{split_name} complete: {len(stem_list)} images and labels.')

class_names = [
    'grub',
    'mole cricket',
    'red spider',
    'aphids',
    'Thrips',
    'Pseudococcus comstocki Kuwana',
    'Trialeurodes vaporariorum',
    'Icerya purchasi Maskell',
    'Ceroplastes rubens',
    'Nipaecoccus vastalor',
    'Scirtothrips dorsalis Hood',
    'Cicadellidae'
]

formatted_names = ', '.join([chr(34) + name + chr(34) for name in class_names])
yaml_lines = [
    'path: E:/cardamom_pest_train/dataset',
    'train: train/images',
    'val: val/images',
    'test: test/images',
    '',
    'nc: 12',
    'names: [' + formatted_names + ']',
    ''
]
yaml_content = chr(10).join(yaml_lines)

yaml_path = os.path.join(dataset_dir, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print('data.yaml generated successfully!')