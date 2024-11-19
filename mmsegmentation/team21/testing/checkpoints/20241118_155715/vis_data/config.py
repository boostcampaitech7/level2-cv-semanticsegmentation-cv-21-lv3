data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        0.0,
        0.0,
        0.0,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        255.0,
        255.0,
        255.0,
    ],
    type='SegDataPreProcessor')
data_root = '/home/minipin/level2_cv_semanticsegmentation-cv-03/data/train'
dataset_type = 'XRayboneDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='mDice',
        type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exp_name = '[test]ExpName'
init_kwargs = dict(
    entity='ppp6131-yonsei-university',
    name='[test]ExpName',
    project='Semantic_segmentation')
launcher = 'none'
load_from = None
log_code_name = 'test-ExpName'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_iters = 20000
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=29,
        type='SegformerHeadWithoutAccuracy'),
    init_cfg=dict(
        checkpoint=
        'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth',
        type='Pretrained'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoderWithoutArgmax')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=6e-05, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=
        '/home/minipin/level2_cv_semanticsegmentation-cv-03/data/train',
        is_train=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadXRayBoneAnnotation'),
            dict(type='TransposeAnnotation'),
            dict(type='PackSegInputs'),
        ],
        type='XRayboneDataset'),
    num_workers=0,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='DiceMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadXRayBoneAnnotation'),
    dict(type='TransposeAnnotation'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=4, type='IterBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=
        '/home/minipin/level2_cv_semanticsegmentation-cv-03/data/train',
        is_train=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayBoneAnnotation'),
            dict(scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='TransposeAnnotation'),
            dict(type='PackSegInputs'),
        ],
        type='XRayboneDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadXRayBoneAnnotation'),
    dict(scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='TransposeAnnotation'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=
        '/home/minipin/level2_cv_semanticsegmentation-cv-03/data/train',
        is_train=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadXRayBoneAnnotation'),
            dict(type='TransposeAnnotation'),
            dict(type='PackSegInputs'),
        ],
        type='XRayboneDataset'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='DiceMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            entity='ppp6131-yonsei-university',
            name='team21/testing',
            project='Semantic_segmentation'),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                entity='ppp6131-yonsei-university',
                name='team21/testing',
                project='Semantic_segmentation'),
            type='WandbVisBackend'),
    ])
wandb_kwargs = dict(
    init_kwargs=dict(
        entity='ppp6131-yonsei-university',
        name='team21/testing',
        project='Semantic_segmentation'))
work_dir = 'team21/testing/checkpoints'
