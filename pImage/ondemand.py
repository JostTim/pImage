# -*- coding: utf-8 -*-

import numpy as np


class OnDemandArray(np.ndarray):
    """This class can be called to create an object accessible like an array to generate frames on demand, in a low ram
    intensive mode but high disk usage mode.
    It will slow down the processes that use the array, but make some os these processes possible that are not possible
    without using huge amounts of ram.
    due to very large amount of data or very high number or arrays loaded in ram at the same time)


    The only rule is that the generator parent class must implements a `get_frame` method:

        class _DelayedData:

        def __init__(self,args,parent):
            self.var1 = args[0]
            self.var2 = args[0]
            ...
            self.parent = parent

        def get_frame(self,index):
            return self.parent.data_accessing_method(arg1, index)
    """

    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, generator_parent=None):

        if generator_parent is None:
            raise ValueError(
                "A DelayedArray must be provided a generator_parent at creation to be able to get image "
                "array frames later on"
            )

        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)

        obj.generator_parent = generator_parent
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.generator_parent = getattr(obj, "generator_parent", None)

    def get_frame(self, index):

        def slice_to_range(slc):
            start = 0 if slc.start is None else slc.start
            stop = self.shape[0] if slc.stop is None else slc.stop
            step = 1 if slc.step is None else slc.step
            return start, stop, step

        if isinstance(index, slice):
            # raise NotImplementedError("A DelayedFramesArray cannot be iterated over with a slice.
            # Must specify a single frame to access.")
            rangeitem = slice_to_range(index)
            for i in range(*rangeitem):
                yield self.generator_parent.get_frame(i).squeeze()

        else:
            if index < 0:
                index = self.shape[0] + index
            yield self.generator_parent.get_frame(index).squeeze()

    def __getitem__(self, index):
        """A delayed Frame array can be indexed with frame id (single index), or in a standard manner"""

        if isinstance(index, tuple):
            frame_index = index[0]
            subslice = index[1:]
        elif isinstance(index, (int, slice)):
            frame_index = index
            subslice = False
        else:
            raise ValueError

        obj = np.array(list(self.get_frame(frame_index))).squeeze()

        if len(obj.shape) > 2:
            return obj[(slice(None), *subslice)] if subslice else obj
        return obj[subslice] if subslice else obj
