# dataset settings
dataset_type = "XRayboneDataset"
data_root = "/home/minipin/level2_cv_semanticsegmentation-cv-03/data/train"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadXRayBoneAnnotation"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="TransposeAnnotation"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadXRayBoneAnnotation"),
    dict(type="TransposeAnnotation"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, is_train=True, pipeline=train_pipeline
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, is_train=False, pipeline=test_pipeline
    ),
)

# 형식만 필요, 값들은 사실 무의미
# Don't fix
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, is_train=False, pipeline=test_pipeline
    ),
)

val_evaluator = dict(type="DiceMetric")
test_evaluator = val_evaluator
