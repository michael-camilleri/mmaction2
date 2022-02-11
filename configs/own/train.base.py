"""
This config is used to train an LFB model.

It should be used in conjunction with a `feature_bank`-type model (e.g. backbone.base.pth): before
using it, one is expected to have generated the feature-bank using feature_bank.base.py.

The script is modified from the [original](https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) with the minimal amount of changes to work with my Data Structure i.e.:
  1. Changes to Paths
  2. Change to Heads to support my limited class list
  3. Removed flipping as an augmentation
  4. Reduced Videos per gpu to 4 (to avoid memory issues)
  5. Reduced Learning Rate and number of Epochs
  6. Set multilabel=False and associated setup


**N.B.**: To Use it, you must search and replace some variables as per below

Usage (from within MMAction2 directory):
  python tools/train.py <path/to/this/config> --validate --seed 0 --deterministic
"""

################## CONFIG VALUES ##################
# Base Config
_base_ = ['backbone.base.py']

# Paths Config
Source_Root = '<SOURCE>'  # Path to root source of data
Feature_Path = '<FEATUREBANK>'  # Path to where Feature-Bank is stored
Model_Path = '<MODELINIT>' # Initial Model
Output_Path = '<MODELOUT>'   # Working Directory
Frames_Offset = 'Frames'
Annotation_File = 'Behaviours.csv'
Label_File = 'Actions.pbtxt'
Detections_File = 'Detections.pkl'

# Other Config
DataSet_Modes = ('Train', 'Validate')
ImageNormalisation = dict(
    mean=[69.199, 69.199, 69.199],
    std=[58.567, 58.567, 58.567],
    to_bgr=False
) # Formerly mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
max_num_sampled_feat = 5
window_size = 60
lfb_channels = 2048

################## STANDARD VALUES ##################

# Model Head Override
model = dict(
    roi_head=dict(
        shared_head=dict(
            type='FBOHead',
            lfb_cfg=dict(
                lfb_prefix_path=Feature_Path,
                max_num_sampled_feat=max_num_sampled_feat,
                window_size=window_size,
                lfb_channels=lfb_channels,
                dataset_modes=DataSet_Modes,
                device='gpu'),
            fbo_cfg=dict(
                type='non_local',
                st_feat_channels=2048,
                lt_feat_channels=lfb_channels,
                latent_channels=512,
                num_st_feat=1,
                num_lt_feat=window_size * max_num_sampled_feat,
                num_non_local_layers=2,
                st_feat_dropout_ratio=0.2,
                lt_feat_dropout_ratio=0.2,
                pre_activate=True)),
        bbox_head=dict(
                in_channels=2560,
                num_classes=9,   # Changed from 81
                multilabel=False, # Enforce single-label
        )
    ),
    test_cfg=dict(rcnn=dict(action_thr=-1))
)


# Training (and Validation) Pipelines
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=16),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Normalize', **ImageNormalisation),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')), # Rename is needed to use mmdet dets
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids', 'img_key'])
]

val_pipeline = [ # The testing is w/o. any cropping / flipping
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=16, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **ImageNormalisation),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')), # Rename is needed to use mmdet dets
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape', 'img_key'],
        nested=True)
]

# Data Definition
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='AVADataset',
        num_classes=9,
        fps=25,
        timestamp_start=0,
        timestamp_end=119,
        ann_file=f'{Source_Root}/Train/{Annotation_File}',
        exclude_file=None,
        pipeline=train_pipeline,
        label_file=f'{Source_Root}/Train/{Label_File}',
        proposal_file=f'{Source_Root}/Train/{Detections_File}',
        person_det_score_thr=0.5,
        data_prefix=f'{Source_Root}/Train/{Frames_Offset}'
    ),
    val=dict(
        type='AVADataset',
        num_classes=9,
        fps=25,
        timestamp_start=0,
        timestamp_end=119,
        ann_file=f'{Source_Root}/Validate/{Annotation_File}',
        exclude_file=None,
        pipeline=val_pipeline,
        label_file=f'{Source_Root}/Validate/{Label_File}',
        proposal_file=f'{Source_Root}/Validate/{Detections_File}',
        person_det_score_thr=0.5,
        data_prefix=f'{Source_Root}/Validate/{Frames_Offset}'
    ),
)
# data['test'] = data['val']
evaluation = dict(interval=1, save_best='mAP@0.5IOU')

# Training Schedule
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-05) # Modified LR
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
total_epochs = 5
checkpoint_config = dict(interval=5)


# Workflow
workflow = [('train', 1)]
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = Output_Path
load_from = Model_Path
resume_from = None
find_unused_parameters = False
