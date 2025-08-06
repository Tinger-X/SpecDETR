dataset_type = 'HSIDataset'
data_root = 'S:/HSI/SpecDETR/SanDiego/'
normalized_basis = 10000
backend_args = None
train_pipeline = [
    dict(
        type='LoadHyperspectralImageFromFiles',
        to_float32=True,
        normalized_basis=10000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='HSIResize', scale_factor=1, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'scale_factor'))
]
test_pipeline = [
    dict(
        type='LoadHyperspectralImageFromFiles',
        to_float32=True,
        normalized_basis=10000),
    dict(type='HSIResize', scale_factor=1, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='HSIDataset',
        data_root='S:/HSI/SpecDETR/SanDiego/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadHyperspectralImageFromFiles',
                to_float32=True,
                normalized_basis=10000),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='HSIResize', scale_factor=1, keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction', 'scale_factor'))
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIDataset',
        data_root='S:/HSI/SpecDETR/SanDiego/',
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadHyperspectralImageFromFiles',
                to_float32=True,
                normalized_basis=10000),
            dict(type='HSIResize', scale_factor=1, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIDataset',
        data_root='S:/HSI/SpecDETR/SanDiego/',
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadHyperspectralImageFromFiles',
                to_float32=True,
                normalized_basis=10000),
            dict(type='HSIResize', scale_factor=1, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='S:/HSI/SpecDETR/SanDiego/annotations/test.json',
    metric=['bbox', 'proposal_fast'],
    classwise=True,
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='S:/HSI/SpecDETR/SanDiego/annotations/test.json',
    metric=['bbox', 'proposal_fast'],
    classwise=True,
    format_only=False,
    backend_args=None)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=999999, by_epoch=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = './work_dirs/SpecDETR/SpecDETR_Sandiego_12e.pth'
resume = False
norm = 'LN'
num_levels = 2
in_channels = 180
embed_dims = 256
query_initial = 'one'
model = dict(
    type='SpecDetr',
    num_queries=900,
    num_query_per_cat=5,
    num_fix_query=0,
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=2,
    candidate_bboxes_size=0.01,
    scale_gt_bboxes_size=0,
    training_dn=True,
    dn_type='CDN',
    query_initial='one',
    remove_last_candidate=False,
    data_preprocessor=dict(type='HSIDetDataPreprocessor'),
    backbone=dict(
        type='No_backbone_ST',
        in_channels=180,
        embed_dims=256,
        patch_size=(1, ),
        num_levels=2,
        norm_cfg=dict(type='LN')),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=2, num_points=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            norm_cfg=dict(type='LN'))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=2, num_points=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            norm_cfg=dict(type='LN')),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='SpecDetrHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        pre_bboxes_round=True,
        use_nms=True,
        iou_threshold=0.01,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.5,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=200)),
    train_cfg=dict(
        assigner=dict(
            type='DynamicIOUHungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(type='IoULossCost', iou_mode='iou', weight=1.0)
            ],
            match_num=10,
            base_match_num=1,
            iou_loss_th=0.05,
            dynamic_match=True)),
    test_cfg=dict(max_per_img=300))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[10],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=4)
launcher = 'none'
work_dir = './work_dirs/SpecDETR/Sandiego'
