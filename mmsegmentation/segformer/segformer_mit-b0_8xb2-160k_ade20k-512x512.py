_base_ = [
    './segformer_mit-b0.py',
    './xraydata2.py',
    './default_runtime.py', './schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=29))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# mixed precision 설정 추가
fp16 = dict(loss_scale=512.)
train_dataloader = dict(batch_size=1, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
