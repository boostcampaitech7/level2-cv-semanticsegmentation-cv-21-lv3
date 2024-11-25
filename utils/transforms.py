import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(is_train=True):
    return A.Compose(
        [
            *( 
                # is_train = True일때 적용되는 증강
                [A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.5),
                    A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8))]
                if is_train else []
            ),
            # 공통으로 들어가는 Transform
            A.Resize(512, 512),
            A.ToFloat(max_value=255), # Normalize
            ToTensorV2(transpose_mask=True)
        ]
    )

