# vim: expandtab:ts=4:sw=4
import time
import collections
import numpy as np
from . import util

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, detection, t_obs, color=None, n_init=3, max_age=30):
        # params
        self._n_init = n_init
        self._max_age = max_age

        # initial position
        self.mean = mean
        self.covariance = covariance
        self.ndim = len(self.mean)//2

        # initial state
        self.state = TrackState.Tentative
        self.track_id = track_id
        # initial time
        self.first_seen = self.last_seen = t_obs
        self.last_predict_time = t_obs
        self.steps_since_update = 0
        self.hits = 1

        # generate a unique color for the track
        self.color = np.random.uniform(0, 1, size=3) if color is None else color

        # tracking detection history
        self.meta = []
        self.features = []
        if detection is not None:
            self.meta.append(detection.meta)
            self.features.append(detection.feature)

    @property
    def age(self):
        return self.last_predict_time - self.first_seen

    @property
    def time_since_update(self):
        return self.last_predict_time - self.last_seen

    @property
    def xyah(self):
        return self.mean[:self.ndim].copy()

    @property
    def tlwh(self):
        return util.xyah2tlwh(self.xyah)

    @property
    def tlbr(self):
        return util.xyah2tlbr(self.xyah)
    
    @property
    def xywh(self):
        return util.xyah2xywh(self.xyah)

    def predict(self, kf, t_obs):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        t_obs = t_obs or time.time()
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance, 
            t_obs - self.last_predict_time)
        self.steps_since_update += 1
        self.last_predict_time = t_obs

    def update(self, kf, detection, t_obs):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.xyah)
        self.features.append(detection.feature)
        self.meta.append(detection.meta)

        self.hits += 1
        self.steps_since_update = 0
        self.last_seen = t_obs
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.steps_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
