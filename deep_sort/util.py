


def get_ndim(xs):
    return xs.shape[-1]//2

def tlbr2tlwh(tlbr):
    """(top left, bottom right) => (top left x, top left y, width, height)"""
    ndim = get_ndim(tlbr)
    tlbr[...,ndim:] -= tlbr[...,:ndim]
    return tlbr

def tlwh2tlbr(tlwh):
    """(top left x, top left y, width, height) => (top left, bottom right)"""
    ndim = get_ndim(tlwh)
    tlwh[...,ndim:] += tlwh[...,:ndim]
    return tlwh

def tlwh2xyah(tlwh):
    """(top left x, top left y, width, height) => (center x, center y, width/height, height)"""
    ndim = get_ndim(tlwh)
    tlwh[...,:ndim] += tlwh[...,ndim:] / 2
    tlwh[...,ndim] /= tlwh[...,-1]
    return tlwh

def xyah2tlwh(xyah):
    """(center x, center y, width/height, height) => (top left x, top left y, width, height)"""
    ndim = get_ndim(xyah)
    xyah[...,ndim] *= xyah[...,-1]
    xyah[...,:ndim] -= xyah[...,ndim:] / 2
    return xyah

def xyah2tlbr(xyah):
    '''(center x, center y, width/height, height) => (top left, bottom right)'''
    return tlwh2tlbr(xyah2tlwh(xyah))

def tlbr2xyah(xyah):
    '''(top left, bottom right) => (center x, center y, width/height, height)'''
    return tlwh2xyah(tlbr2tlwh(xyah))