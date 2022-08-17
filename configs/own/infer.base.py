"""
This config is used to infer from an LFB model.

It should be used in conjunction with a `feature_bank`-type model (e.g.
backbone.base.pth): before using it, one is expected to have generated the feature-bank
using feature_bank.base.py.

The script is a rehash of train.base.py, focusing on a Testing Set

Usage (from within MMAction2 directory):
  python tools/test.py <path/to/this/config> <path/to/model/pth> --out results.csv
"""

################## CONFIG VALUES ##################
# Base Config
_base_ = ['backbone.base.py']

# Paths Config
Source_Root = '<SOURCE>'  # Path to root source of data
Feature_Path = '<FEATUREBANK>'  # Path to where Feature-Bank is stored
Output_Path = '<RESULTS>'   # Working Directory
Frames_Offset = '<FRAMES>'
Image_Template = '<IMAGE_TEMPLATE>'
Annotation_File = 'AVA.Behaviours.csv'
Label_File = 'AVA.Actions.pbtxt'
Detections_File = 'AVA.Detections.pkl'

# Other Config
DataSet = '<DATASET>' # [Test|Validate]
ImageNormalisation = dict(
    mean=[69.201, 69.201, 69.201],
    std=[58.571, 58.571, 58.571],
    to_bgr=False
) # Formerly mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
max_num_sampled_feat = 5
window_size = 60
lfb_channels = 2048

################## STANDARD VALUES ##################

# Model Head Override
#  Note that at Test Time, we are always multi-label to output all probabilities.
model = dict(
    roi_head=dict(
        shared_head=dict(
            type='FBOHead',
            lfb_cfg=dict(
                lfb_prefix_path=Feature_Path,
                max_num_sampled_feat=max_num_sampled_feat,
                window_size=window_size,
                lfb_channels=lfb_channels,
                dataset_modes=(DataSet,),
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
    test_cfg=dict(rcnn=dict(action_thr=0.0)) # Changed to 0 to give out all values.
)


# Testing Pipeline
infer_pipeline = [ # The testing is w/o. any cropping / flipping
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
    videos_per_gpu=1,
    workers_per_gpu=2,    
    test=dict(
        type='AVADataset',
        num_classes=9,
        fps=25,
        start_index=0,
        timestamp_start=0,
        timestamp_end=<NUM_BTIS>,
        filename_tmpl=Image_Template,
        ann_file=f'{Source_Root}/{DataSet}/{Annotation_File}',
        exclude_file=None,
        pipeline=infer_pipeline,
        label_file=f'{Source_Root}/{Label_File}',
        proposal_file=f'{Source_Root}/{DataSet}/{Detections_File}',
        person_det_score_thr=0.5,
        data_prefix=f'{Source_Root}/{Frames_Offset}'
    ),
)

dist_params = dict(backend='nccl')
work_dir = Output_Path
