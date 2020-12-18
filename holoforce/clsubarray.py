import pyopencl as cl
import pyopencl.array as cla
import numpy as np


def _unravel(x, strides):
    for s in strides:
        s, x = divmod(x, s)
        yield s


def get_rect_params_np(a):
    # bytesbase = a.base.view(dtype=np.uint8)
    # requirement: along fastest varying axes data is contiguous in memory
    # a is rectangular view (same dimensionality) into c contiguous memory

    # TODO####align by strides
    base = a if a.base is None else a.base

    assert a.ndim == base.ndim  # TODO: nur Vorsichtsmaßnahme
    assert a.strides[-1] == a.itemsize, 'wrong memory layout'

    # shape of _bytes_ view into a.base (but only based on a and assumptions)
    bytesshape = list(a.shape)
    bytesshape[-1] *= a.itemsize

    bytesstrides = list(a.strides)
    bytesstrides[-1] = 1

    addr, foo = a.__array_interface__['data']
    addr_base, foo = base.__array_interface__['data']
    offset = addr - addr_base

    bytesorigin = list(_unravel(offset, bytesstrides))

    return tuple(bytesorigin[::-1]), tuple(bytesshape[::-1]), tuple(bytesstrides[-2::-1])


def get_rect_params_cl(a : cla.Array):
    # assert a.ndim == base.ndim
    assert a.strides[-1] == a.dtype.itemsize, 'wrong memory layout'

    # shape of _bytes_ view into a.base (but only based on a and assumptions)
    bytesshape = list(a.shape)
    bytesshape[-1] *= a.dtype.itemsize

    bytesstrides = list(a.strides)
    bytesstrides[-1] = 1

    offset = a.offset

    bytesorigin = list(_unravel(offset, bytesstrides))

    # origin, region, pitches
    return tuple(bytesorigin[::-1]), tuple(bytesshape[::-1]), tuple(bytesstrides[-2::-1])


def get_rect_kwargs(host, buffer):
    host_shape = list(host.shape)
    host_strides = list(host.strides)
    buffer_shape = list(buffer.shape)
    buffer_strides = list(buffer.strides)

    if (host_strides[-1] != host.itemsize
            or buffer_strides[-1] != buffer.dtype.itemsize
            and host.ndim < 3
            and buffer.ndim < 3):

        host_shape.append(1)
        host_strides.append(host.itemsize)
        buffer_shape.append(1)
        buffer_strides.append(buffer.dtype.itemsize)

    host_shape[-1] *= host.itemsize
    buffer_shape[-1] *= buffer.dtype.itemsize

    host_strides[-1] = 1
    buffer_strides[-1] = 1

    #base = host if host.base is None else host.base
    #base = host.base or host
    host_origin = list(_unravel(host.__array_interface__['data'][0] - (host.base or host).__array_interface__['data'][0],
                          host_strides))
    buffer_origin = list(_unravel(buffer.offset,
                               buffer_strides))

    assert host_shape == buffer_shape, 'shape does not match'
    result = dict(region = host_shape[::-1],
                  host_origin=host_origin[::-1],
                  buffer_origin=buffer_origin[::-1],
                  host_pitches=host_strides[-2::-1],
                  buffer_pitches=buffer_strides[-2::-1])
    return result


class SubArray(cla.Array):

    def set(self, ary, queue=None, async_=None):
        assert ary.size == self.size, "size does not match"
        assert ary.dtype == self.dtype, 'type does not match'

        if cla._equal_strides(ary.strides, self.strides, self.shape):
            evt = cl.enqueue_copy(queue or self.queue,
                                  self.base_data, ary,
                                  device_offset=self.offset,
                                  is_blocking=not async_)
            self.add_event(evt)
        else:
            kwargs = get_rect_kwargs(ary, self)
            evt = cl._enqueue_write_buffer_rect(queue or self.queue,
                                          mem=self.base_data,
                                          hostbuf=ary,
                                          is_blocking=not async_,
                                          **kwargs,
                                          )
            self.add_event(evt)

    def get(self, queue=None, ary=None, async_=None):
        if ary is None:
            ary = np.empty(self.shape, self.dtype)
        else:
            assert ary.size == self.size, "size does not match"
            assert ary.dtype == self.dtype, "type does not match"

        if cla._equal_strides(ary.strides, self.strides, self.shape):
            cl._enqueue_read_buffer(queue or self.queue,
                                mem=self.base_data, hostbuf=ary,
                                device_offset=self.offset,
                                wait_for=self.events, is_blocking=not async_)
        else:
            kwargs = get_rect_kwargs(ary, self)
            cl._enqueue_read_buffer_rect(queue or self.queue,
                                         mem=self.base_data,
                                         hostbuf=ary,
                                         is_blocking=not async_,
                                         wait_for=self.events,
                                         **kwargs)
        return ary








