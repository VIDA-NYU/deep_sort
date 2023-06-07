# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, ndim=2):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter(ndim=ndim)
        self.tracks = {}
        self._next_id = 1

    def step(self, detections, t_obs):
        """Propagate track state distributions one time step forward and perform a state update.
        """
        self.predict(t_obs)
        self.update(detections, t_obs)

    def predict(self, t_obs):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks.values():
            track.predict(self.kf, t_obs)

    def update(self, detections, t_obs):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade - match detections to tracks
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        self._update(detections, t_obs, matches, unmatched_tracks, unmatched_detections)

    def _get_unmatched(self, detections, matches):
        track_ids = set(self.tracks)
        matched_track_ids, matched_detections = tuple(zip(*matches)) or ((), ())
        unmatched_tracks = track_ids - set(matched_track_ids)
        unmatched_detections = set(range(len(detections))) - set(matched_detections)
        return unmatched_tracks, unmatched_detections

    def _update(self, detections, t_obs, matches, unmatched_tracks=None, unmatched_detections=None):
        # Update matched tracks
        for track_idx, detection_idx in matches:
            if track_idx not in self.tracks:
                self._initiate_track(detections[detection_idx], t_obs, track_idx)
            else:
                self.tracks[track_idx].update(self.kf, detections[detection_idx], t_obs)

        # auto-detect missed track ids and detections
        matched_track_ids, matched_detections = tuple(zip(*matches)) or ((), ())
        if unmatched_detections is None:
            unmatched_detections = set(range(len(detections))) - set(matched_detections)
        if unmatched_tracks is None:
            unmatched_tracks = set(self.tracks) - set(matched_track_ids)

        # missed tracks
        for track_idx in unmatched_tracks or []:
            self.tracks[track_idx].mark_missed()

        # new detections
        for detection_idx in unmatched_detections or []:
            self._initiate_track(detections[detection_idx], t_obs)

        # deleted tracks
        deleted = {
            tid: self.tracks.pop(tid) 
            for tid in set(self.tracks) 
            if self.tracks[tid].is_deleted()
        }

        # update nearest neighbors using features
        self._fit()
        return deleted

    # def _delete_tracks(self, tracks):
    #     if hasattr(self.metric, 'delete_targets'):
    #         self.metric.delete_targets([t.track_id for t in tracks])

    def _gated_metric(self, tracks, dets, track_indices, detection_indices):
        '''Match based on appearance features.'''
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = linear_assignment.gate_cost_matrix(
            self.kf, cost_matrix, tracks, dets, track_indices,
            detection_indices)
        return cost_matrix
    
    def gated_metric(self, dets, track_indices=None):
        confirmed_tracks = [i for i, t in self.tracks.items() if t.is_confirmed()]
        track_ids = track_indices or list(confirmed_tracks)
        cost = self._gated_metric(self.tracks, dets, track_ids, range(len(dets)))
        return cost, track_ids

    def _match(self, detections):
        # match confirmed tracks based on appearance, 
        # match remaining unmatched/confirmed tracks with IoU

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks.values()) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks.values()) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                self._gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks.values(), detections, confirmed_tracks)

        # get unmatched tracks that were visible in the last frame to do IoU comparison
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].steps_since_update <= 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].steps_since_update > 1]

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks.values(),
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _fit(self):
        # collect track features
        confirmed_tracks = [t for t in self.tracks.values() if t.is_confirmed()]
        active_targets = [t.track_id for t in confirmed_tracks]
        features, targets = [], []
        for track in confirmed_tracks:
            features.extend(track.features)
            targets.extend(track.track_id for _ in track.features)
            track.features = []

        # Update distance metric.
        self.metric.partial_fit(
            np.asarray(features), 
            np.asarray(targets), 
            np.asarray(active_targets))

    def _initiate_track(self, detection, t_obs, track_id=None):
        # create new track
        mean, covariance = self.kf.initiate(detection.xyah)
        if track_id is None:
            track_id = self._next_id
        self.tracks[track_id] = Track(
            mean, covariance, track_id, 
            detection=detection,
            t_obs=t_obs, 
            n_init=self.n_init, 
            max_age=self.max_age)
        self._next_id = max(self.tracks) + 1
