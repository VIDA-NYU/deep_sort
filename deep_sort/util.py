


def get_ndim(xs):
    return xs.shape[-1]//2

# tlwh <-> tlbr

def tlwh2tlbr(tlwh, ndim=None):
    """(top left x, top left y, width, height) => (top left, bottom right)"""
    ndim = ndim or get_ndim(tlwh)
    tlwh[...,ndim:] += tlwh[...,:ndim]
    return tlwh


def tlbr2tlwh(tlbr, ndim=None):
    """(top left, bottom right) => (top left x, top left y, width, height)"""
    ndim = ndim or get_ndim(tlbr)
    tlbr[...,ndim:] -= tlbr[...,:ndim]
    return tlbr

# tlwh <-> xywh

def tlwh2xywh(tlwh, ndim=None):
    """(top left x, top left y, width, height) => (top left, bottom right)"""
    ndim = ndim or get_ndim(tlwh)
    tlwh[...,:ndim] += tlwh[...,ndim:] / 2
    return tlwh

def xywh2tlwh(xywh, ndim=None):
    """(top left, bottom right) => (top left x, top left y, width, height)"""
    ndim = ndim or get_ndim(xywh)
    xywh[...,:ndim] -= xywh[...,ndim:] / 2
    return xywh

# tlwh <-> xyah

def tlwh2xyah(tlwh, ndim=None):
    """(top left x, top left y, width, height) => (center x, center y, width/height, height)"""
    return xywh2xyah(tlwh2xywh(tlwh, ndim), ndim)

def xyah2tlwh(xyah, ndim=None):
    """(center x, center y, width/height, height) => (top left x, top left y, width, height)"""
    return xywh2tlwh(xyah2xywh(xyah, ndim), ndim)

# xywh <-> xyah

def xywh2xyah(xywh, ndim=None):
    """(top left x, top left y, width, height) => (center x, center y, width/height, height)"""
    ndim = ndim or get_ndim(xywh)
    xywh[...,ndim] /= xywh[...,ndim+1]
    if ndim == 3:
        xywh[...,-1] /= xywh[...,ndim+1]
    return xywh

def xyah2xywh(xyah, ndim=None):
    """(center x, center y, width/height, height) => (top left x, top left y, width, height)"""
    ndim = ndim or get_ndim(xyah)
    xyah[...,ndim] *= xyah[...,ndim+1]
    if ndim == 3:
        xyah[...,-1] *= xyah[...,ndim+1]
    return xyah

# xywh <-> tlbr

def xywh2tlbr(xywh, ndim=None):
    '''(center x, center y, width/height, height) => (top left, bottom right)'''
    return tlwh2tlbr(xywh2tlwh(xywh, ndim), ndim)

def tlbr2xwh(tlbr, ndim=None):
    '''(top left, bottom right) => (center x, center y, width/height, height)'''
    return tlwh2xywh(tlbr2tlwh(tlbr, ndim), ndim)

# xyah <-> tlbr

def xyah2tlbr(xyah, ndim=None):
    '''(center x, center y, width/height, height) => (top left, bottom right)'''
    return tlwh2tlbr(xyah2tlwh(xyah, ndim), ndim)

def tlbr2xyah(xyah, ndim=None):
    '''(top left, bottom right) => (center x, center y, width/height, height)'''
    return tlwh2xyah(tlbr2tlwh(xyah, ndim), ndim)
