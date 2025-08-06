# dataset settings
dataset_type = 'HSIDataset'
# Avon, MUUFLGulfport, SanDiego, SPOD_30b_8c
data_root = 'S:/HSI/SpecDETR/Avon/'

normalized_basis = 3000
backend_args = None
train_pipeline = [
    dict(type='LoadHyperspectralImageFromFiles', to_float32=True, normalized_basis=normalized_basis),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='HSIResize', scale_factor=1, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadHyperspectralImageFromFiles', to_float32=True, normalized_basis=normalized_basis),
    # dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='HSIResize', scale_factor=1, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'proposal_fast'],
    classwise=True,
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


#################################################################################################################

# # dataset settings
# dataset_type = 'HSIDataset'
# data_root = 'S:/HSI/SpecDETR/SPOD_30b_8c/'
#
# normalized_basis = 3000
# backend_args = None
# train_pipeline = [
#     dict(type='LoadHyperspectralImageFromFiles', to_float32=True, normalized_basis=normalized_basis),
#     dict(type='LoadAnnotations', with_bbox=True),
#     # dict(type='Resize', scale=(512, 512), keep_ratio=True),
#     dict(type='HSIResize', scale_factor=1, keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction', 'scale_factor'))
# ]
# test_pipeline = [
#     dict(type='LoadHyperspectralImageFromFiles', to_float32=True, normalized_basis=normalized_basis),
#     # dict(type='Resize', scale=(512, 512), keep_ratio=True),
#     dict(type='HSIResize', scale_factor=1, keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
# ]
# train_dataloader = dict(
#     batch_size=4,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/test.json',
#         data_prefix=dict(img='test/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/train.json',
#         data_prefix=dict(img='train/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))
# test_dataloader = val_dataloader
#
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/train.json',
#     metric=['bbox', 'proposal_fast'],
#     classwise=True,
#     format_only=False,
#     backend_args=backend_args)
# test_evaluator = val_evaluator
