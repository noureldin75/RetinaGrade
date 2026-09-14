import numpy as np
import albumentations as A
import cv2
def count_labels(y_train):
    return np.unique(y_train, return_counts=True) #_,counts


def over_sample(y_train,targets={1: 2,2: 1, 3: 5, 4: 3}):
    final_indices=[np.arange(len(y_train))]
    for label, multiplier in targets.items():
        label_indices = np.where(y_train == label)[0]
        oversampled_indices = np.tile(label_indices, multiplier)
        final_indices.append(oversampled_indices)
    return np.concatenate(final_indices)


train_transform = A.Compose([
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(-0.2, 0.2),
        rotate=(-180, 180),
        shear= (-0.2, 0.2),
        p=0.5,
    ),

    A.HorizontalFlip(p=0.5),


    A.RandomBrightnessContrast(
        brightness_limit=20/255.0, 
        contrast_limit=0.2, 
        p=0.5
        ),

    A.HueSaturationValue(
        hue_shift_limit=10, 
        sat_shift_limit=20, 
        val_shift_limit=0, 
        p=0.5
    ),

    A.OneOf([
        A.Blur(blur_limit=3, p=1.0),
        A.Sharpen(alpha_limit=0.2, p=1.0),
    ],
    p=0.5
    ),
        
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
])


val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
])

def apply_transform(transform, image):
    augmented = transform(image=image)
    return augmented['image']




#V1 augmentation
# A.HorizontalFlip(p=0.5),
# A.OneOf([
#     A.RandomRotate90(p=0.5),
#     A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
# ], p=0.7),
# A.RandomBrightnessContrast(p=0.1),
# A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# A.ToTensorV2()