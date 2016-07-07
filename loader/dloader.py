# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import ctypes as ct
import os
import atexit


class DataLoader(object):

    """
    Encapsulates the data loader library and exposes an API to iterate over
    generic data (images, video or audio given in compressed form). An index
    file that maps the data examples to their targets is expected to be provided
    in CSV format.

    Arguments:
        loader_cfg_string (json encoded str):
            All configuration information needed for defining the type of extraction,
            transforming, and loading of targets, as well as where source files come from,
            where the files should be cached locally, etc.
        batch_size (int):
            The number of records to return per batch (minibatch)
        backend (object):
            This is an instance of an object which knows how to create tensors it needs
            for processing, and how to transfer host information to those tensors
    """

    def __init__(self, loader_cfg_string, batch_size, backend):
        self.batch_size = batch_size

        self.item_count = ct.c_int(0)
        self.buffer_id, self.start_idx = 0, 0

        self._load_library()

        # Launch background threads
        self.loader = self._start(loader_cfg_string, backend)

        atexit.register(self._stop, self)

    def _load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, 'bin', 'loader.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.get_error_message.restype = ct.c_char_p
        self.loaderlib.start.restype = ct.c_void_p

        self.loaderlib.next.argtypes = [ct.c_void_p, ct.c_int]
        self.loaderlib.next.restype = ct.py_object

        self.loaderlib.stop.argtypes = [ct.c_void_p]
        self.loaderlib.reset.argtypes = [ct.c_void_p]
        self.loaderlib.itemCount.argtypes = [ct.c_void_p]
        self.loaderlib.itemCount.restype = ct.c_int

    def _raise_loader_error(self):
        raise RuntimeError(
            'error in loader: {}'.format(
                self.loaderlib.get_error_message()
            )
        )

    def _start(self, loader_cfg_string, backend):
        loader = self.loaderlib.start(
            ct.c_char_p(loader_cfg_string),
            ct.py_object(backend)
        )

        if loader is None:
            self._raise_loader_error()

        return loader

    def _stop(self):
        if self.loaderlib.stop(self.loader) == 1:
            self._raise_loader_error()

    def _itemCount(self):
        itemCount = self.loaderlib.itemCount(self.loader)
        if itemCount == -1:
            self._raise_loader_error()

        return itemCount

    def _next(self, buffer_id):
        tup = self.loaderlib.next(
            self.loader, ct.c_int(buffer_id)
        )

        if tup is None:
            self._raise_loader_error()

        return tup

    def _reset(self):
        if self.loaderlib.reset(self.loader) == -1:
            self._raise_loader_error()

    @property
    def nbatches(self):
        return -((self.start_idx - self.itemCount) // self.batch_size)

    @property
    def itemCount(self):
        return self._itemCount()

    def reset(self):
        """
        Restart data from index 0
        """
        self.buffer_id, self.start_idx = 0, 0
        self._reset()

    def next(self, start):
        end = min(start + self.batch_size, self.itemCount)
        if end == self.itemCount:
            self.start_idx = self.batch_size - (self.itemCount - start)

        (data, targets) = self._next(self.buffer_id)

        # Toggle buffer_id between 0 and 1
        self.buffer_id = 1 - self.buffer_id

        return (data, targets)

    def __iter__(self):
        for start in range(self.start_idx, self.itemCount, self.batch_size):
            yield self.next(start)
