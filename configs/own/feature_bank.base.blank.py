"""
This config is used to generate long-term feature bank.

It has the minimal amount of changes to work with my Data Structure i.e.:
  1. Changes to Paths
  2. Changed Detection Score Threshold (0.5 rather than 0.9)
  3. Image Normalisation (my parameters)
  4. Re-Organisation of the Configurations to be more understandable to me.

**N.B.**: To Use it, you must search and replace for <CHOOSE_DATASET> to set the DataSet name.

It should be used in conjunction with a `feature_bank`-type model (e.g. backbone.base.pth) 

Usage (from within MMAction2 directory):
python tools/test.py <path/to/this/config> <path/to/Pytorch/Model> --out <path/to/output/csv>
"""
################## CONFIG VALUES ##################
# Base Config
_base_ = ['backbone.base.py']

# Paths Config
Source_Root = '/media/veracrypt4/Q1/Snippets/Sample'
Output_Path = '/home/s1238640/Documents/DataSynced/PhD Project/Data/MRC Harwell/Scratch'
Frames_Offset = 'Frames'
Annotation_File = 'Behaviours.csv'
Label_File = 'Actions.pbtxt'
Detections_File = 'Detections.pkl'

# Other Config
# <DATASET> # You need to specify DataSet=[Train|Validate|Test]
HalfPrecision = True # [True|False]
ImageNormalisation = dict(mean=[69.199, 69.199, 69.199], std=[58.567, 58.567, 58.567], to_bgr=False) # Formerly mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]


################## STANDARD VALUES ##################

# Model Head Override
model = dict(
    roi_head=dict(
        shared_head=dict(
            type='LFBInferHead',
            lfb_prefix_path=Output_Path,
            dataset_mode=DataSet,
            use_half_precision=HalfPrecision)))

# Feature-Bank Generation Pipeline
infer_pipeline = [
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=16),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **ImageNormalisation),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),    # Rename is needed to use mmdet detectors
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
    workers_per_gpu=4,
    test=dict(
        type='AVADataset',
        num_classes=9,
        fps=25,
        timestamp_start=0,
        timestamp_end=119,
        ann_file=f'{Source_Root}/{DataSet}/{Annotation_File}',
        exclude_file=None,
        pipeline=infer_pipeline,
        label_file=f'{Source_Root}/{DataSet}/{Label_File}',
        proposal_file=f'{Source_Root}/{DataSet}/{Detections_File}',
        person_det_score_thr=0.5,
        data_prefix=f'{Source_Root}/{DataSet}/{Frames_Offset}'))

# Not Sure (as original)
dist_params = dict(backend='nccl')
work_dir = Output_Path
