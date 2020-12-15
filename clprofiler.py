#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np


class CLProfilerABC(object):
    def __init__(self, name=''):
        pass

    def time(self, event, comment=''):
        pass

    def reset(self):
        pass

    def summary(self, select=''):
        pass

    def __bool__(self):
        """return True if profiler is actively profiling"""
        pass


class NoCLProfiler(CLProfilerABC):
    def __init__(self, name='profiling inactive'):
        self.name = name

    def time(self, event, comment=''):
        return event

    def summary(self, select=''):
        return self.name

    def __bool__(self):
        return False


class CLProfiler(CLProfilerABC):

    def __init__(self, name):
        self.name = name
        self.timings = []

    def time(self, event, comment=''):
        try:
            event.wait()
            dt = 1e-3 * (event.profile.end - event.profile.start)
            t0 = 1e-3 * event.profile.submit
            self.timings.append( (t0, dt, comment) )

        except cl.RuntimeError as e:
            pass

        return event

#    def details(self, select=''):
#        result = []

    def summary(self, select=''):
        t = np.array( [dt for (t0, dt, comment) in self.timings if comment.startswith(select)])

        return "%40s N: %4d, avg: %4.0f µs, med: %4.0f µs, min: %4.0f µs" % (
            ' '.join((self.name, select)),
            len(t),
            np.mean(t) if len(t)>0 else np.NaN,
            np.median(t),
            t.min(initial=np.inf),
        )

    def reset(self):
        self.count = 0
        self.timings = []

    def __bool__(self):
        return True

def create_profiler(profiler=False):
    """create a profiler:

    Arguments:
    ----------
    profiler: (None | False | True ) or profiler
        if None or False, return new NoCLProfiler instance
        if True, return new CLProfiler instance
        if profiler given, return profiler unchanged

    """
    if profiler is None:
        return NoCLProfiler()
    if profiler in (False, True):
        return CLProfiler() if profiler else NoCLProfiler()
    if isinstance(profiler, CLProfilerABC):
        return profiler