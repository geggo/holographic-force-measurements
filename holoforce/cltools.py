# -*- coding: utf-8 -*-

import attr
import pyopencl as cl
import pyopencl.tools
from .clprofiler import create_profiler


@attr.s
class CLConsumer(object):
    cl_context: cl.Context = attr.ib(default=None)
    cl_queue: cl.CommandQueue = attr.ib(default=None)
    cl_profiler = attr.ib(default=None)
    cl_allocator = attr.ib(default=None)

    @cl_context.validator
    def _check_cl_context(self, attribute, value):
        if value is None:
            return self.cl_queue.context

    def __attrs_post_init__(self):
        if self.cl_context is None:
            self.cl_context = self.cl_queue.context

        if self.cl_queue is None:

            queueprops = cl.command_queue_properties.PROFILING_ENABLE if self.cl_profiler else 0
            self.cl_queue = cl.CommandQueue(self.cl_context, properties=queueprops)

        self.cl_profiler = create_profiler(self.cl_profiler)

        ## TODO: always initialized
        #if self.cl_allocator is None:
        #    self.cl_allocator = pyopencl.tools.MemoryPool(pyopencl.tools.ImmediateAllocator(self.cl_queue))
