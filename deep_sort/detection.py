# vim: expandtab:ts=4:sw=4
import numpy as np
from . import util

class Detection:
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature, **meta):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.meta = meta

    @property
    def xyah(self):
        return util.tlwh2xyah(self.tlwh.copy())

    @property
    def tlbr(self):
        return util.tlwh2tlbr(self.tlwh.copy())

    @property
    def xywh(self):
        return util.tlwh2xywh(self.tlwh.copy())