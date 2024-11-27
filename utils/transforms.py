import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(config, is_train=True):
    img_size = config['model'].get('img_size', 512)
    return A.Compose(
        [
            *( 
                # is_train = True일때 적용되는 증강
                [
                #     A.ShiftScaleRotate(
                #     shift_limit=0.0625,
                #     scale_limit=0.1,
                #     rotate_limit=15,
                #     interpolation=1,
                #     border_mode=4,
                #     p=0.7
                # ),
                 A.HorizontalFlip(p=0.5),
        
                # 밝기/대비 조정
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.8
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),
                ], p=0.5),
                
                # 경계 강화
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                    ], p=0.3),
                
                # 기하학적 변환
                # A.OneOf([
                #     A.ElasticTransform(
                #         alpha=120,
                #         sigma=120 * 0.05,
                #         alpha_affine=120 * 0.03,
                #         p=0.5
                #     ),
                #     A.GridDistortion(p=0.5),
                #     A.OpticalDistortion(p=0.5),
                # ], p=0.3),
                ]
                        if is_train else []
                    ),
            # 공통으로 들어가는 Transform
            A.Resize(img_size,img_size),
            A.Normalize(),
            ToTensorV2()
        ],   
    )

