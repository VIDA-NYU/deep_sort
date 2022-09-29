# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment
from . import util

# def fix_box_flip(tlwh, ndim=None):
#     ndim = ndim or util.get_ndim(tlwh)
#     tlwh[...,:ndim] -= np.minimum(0, tlwh[...,ndim:])
#     tlwh[...,ndim:] = np.abs(tlwh[...,ndim:])
#     return tlwh

def iou(tlwh, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    ndim = util.get_ndim(tlwh)
    tlbr = util.tlwh2tlbr(tlwh.copy())
    tlbr_can = util.tlwh2tlbr(candidates.copy())

    area_intersection = np.maximum(
        0., 
        np.maximum(tlbr[ndim:], tlbr_can[:, ndim:]) - 
        np.minimum(tlbr[:ndim], tlbr_can[:, :ndim])
    ).prod(axis=1)
    area_bbox = tlwh[ndim:].prod()
    area_candidates = candidates[:, ndim:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].steps_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].tlwh
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
