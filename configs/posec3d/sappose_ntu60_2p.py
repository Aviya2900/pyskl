# model_cfg
backbone_cfg = dict(
    type='MMPoseConv3D_SAP',
    speed_ratio=1,
    channel_ratio=1,
    feats_detach=True,
    pose_detach=True,
    sampling=True,
    feats_pathway=dict(
        channel_ratio=1,
        speed_ratio=1,
        lateral=True,
        lateral_inv=False,
        lateral_infl=1,
        lateral_activate=[1, 1, 1],
        expansion=[1, 4, 8],
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_kernel=(1, 3, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    pose_pathway=dict(
        channel_ratio=1,
        speed_ratio=1,
        lateral=True,
        lateral_inv=False,
        lateral_infl=1,
        lateral_activate=[1, 1, 1],
        expansion=[1, 4, 8],
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)))
head_cfg = dict(
    type='MMPoseHead',
    num_classes=11,
    in_channels=[512, 512],
    loss_components=['feats', 'pose'],
    loss_weights=[1., 1.])
test_cfg = dict(average_clips='prob')
model = dict(
    type='MMRecognizer3D_SAP',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    test_cfg=test_cfg)

dataset_type = 'PoseDataset_2P'
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
class_prob = [1] * 60 + [2] * 60
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'keypoint', 'label']),
    dict(type='SampleTensor', list=['imgs', 'keypoint']),
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'keypoint']),
    dict(type='SampleTensor', list=['imgs', 'keypoint']),
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'keypoint']),
    dict(type='SampleTensor', list=['imgs', 'keypoint']),
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='xset_train_2p',
            pipeline=train_pipeline,
            class_prob=class_prob)),
    val=dict(type=dataset_type, ann_file=ann_file, split='xset_val_2p', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='xset_val_2p', pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/posec3d/slowonly_r50_ntu120_xsub/joint_2P_SAP'