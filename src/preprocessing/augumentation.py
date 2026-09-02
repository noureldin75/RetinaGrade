import numpy as np
import albumentations as A
import cv2
def count_labels(y_train):
    return np.unique(y_train, return_counts=True) #_,counts


def over_sample(y_train,targets={1: 2, 3: 5, 4: 3}):
    final_indices=[np.arange(len(y_train))]
    for label, multiplier in targets.items():
        label_indices = np.where(y_train == label)[0]
        oversampled_indices = np.tile(label_indices, multiplier)
        final_indices.append(oversampled_indices)
    return np.concatenate(final_indices)


transform = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    ], p=0.7),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
])

def apply_transform(image):
    augmented = transform(image=image)
    return augmented['image']
    
