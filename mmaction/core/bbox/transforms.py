# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def bbox2result(bboxes, labels, num_classes, thr=0.01):
    """
    Convert detection results to a list of numpy arrays.

    This identifies single-label classification (as opposed to multi-label)
    through the thr parameter which is set to a negative value

    TODO: Make more efficient
    Args:
        bboxes (Tensor): shape (n, 4)
        labels (Tensor): shape (n, #num_classes)
        num_classes (int): class number, including background class
        thr (float): The score threshold used when converting predictions to
            detection results. If a single negative value, uses single-label
            classification
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return list(np.zeros((num_classes - 1, 0, 5), dtype=np.float32))

    bboxes = bboxes.cpu().numpy()
    scores = labels.cpu().numpy()  # rename for clarification

    # Although we can handle single-label classification, we still want scores
    assert scores.shape[-1] > 1

    # Robustly check for multi/single-label:
    if not hasattr(thr, '__len__'):
        multilabel = thr >= 0
        thr = (thr,) * num_classes
    else:
        multilabel = True

    # Check Shape
    assert scores.shape[1] == num_classes
    assert len(thr) == num_classes

    result = []
    for i in range(num_classes - 1):
        if multilabel:
            where = (scores[:, i + 1] > thr[i + 1])
        else:
            where = (scores[:, 1:].argmax(axis=1) == i)
        result.append(
            np.concatenate(
                (bboxes[where, :4], scores[where, i + 1:i + 2]),
                axis=1
            )
        )
    return result
