import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(is_train=True, size=512):
    return A.Compose(
        [
            *( 
                # is_train = True일때 적용되는 증강
                [A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8)),
                 A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),  # Edge enhancement
                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                 A.Rotate(limit=10, p=0.5),
                 A.HorizontalFlip(p=0.5),
                 A.CoarseDropout(
                     max_holes=32,  # 마스킹할 최대 영역 개수 (기본값: 8)
                     max_height=32,  # 마스킹할 영역의 최대 높이 (기본값: 8)
                     max_width=32,  # 마스킹할 영역의 최대 너비 (기본값: 8)
                     mask_fill_value=0,  # 마스크에서 마스킹된 영역의 채움 값 (기본값: None, 변경되지 않음)
                     p=0.5,  # 적용 확률 (기본값: 0.5)
                 ),
                ]
                if is_train else []
            ),
            # 공통으로 들어가는 Transform
            A.Resize(size, size),
            A.ToFloat(max_value=255), # Normalize
            ToTensorV2(transpose_mask=True)
        ]
    )

