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
import json


class DataLoader(object):

    """
    Encapsulates the data loader library and exposes an API to iterate over
    generic data (images, video or audio given in compressed form). An index
    file that maps the data examples to their targets is expected to be
    provided in CSV format.

    Arguments:
        config (dict):
            All configuration information needed for defining the type of
            extraction, transforming, and loading of targets, as well as where
            source files come from, where the files should be cached locally,
            etc.
        backend (object):
            This is an instance of an object which knows how to create tensors
            it needs for processing, and how to transfer host information to
            those tensors

    Note that if the epoch is not evenly divisible by the minibatch size, there
    will be one minibatch per epoch (or so) which contains data from two
    epochs.
    """

    def __init__(self, config, backend):
        self.buffer_id = 0

        self._load_library()

        # Launch background threads
        self.loader = self._start(json.dumps(config), backend)

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
        """
        C api can't easily raise python exceptions, so it returns an error code
        and then makes the error message accessable via get_error_message().
        This gets that message and wraps it in a RuntimeError.
        """
        raise RuntimeError(
            'error in loader: {}'.format(
                self.loaderlib.get_error_message()
            )
        )

    def _start(self, config, backend):
        """
        C api wrapper with exception handling
        """
        loader = self.loaderlib.start(
            ct.c_char_p(config),
            ct.py_object(backend)
        )

        if loader is None:
            self._raise_loader_error()

        return loader

    def _stop(self):
        """
        C api wrapper with exception handling
        """
        if self.loaderlib.stop(self.loader) == 1:
            self._raise_loader_error()

    def _itemCount(self):
        """
        C api wrapper with exception handling
        """
        itemCount = self.loaderlib.itemCount(self.loader)
        if itemCount == -1:
            self._raise_loader_error()

        return itemCount

    def _next(self, buffer_id):
        """
        C api wrapper with exception handling
        """
        tup = self.loaderlib.next(
            self.loader, ct.c_int(buffer_id)
        )

        if tup is None:
            self._raise_loader_error()

        return tup

    def _reset(self):
        """
        C api wrapper with exception handling
        """
        if self.loaderlib.reset(self.loader) == -1:
            self._raise_loader_error()

    @property
    def itemCount(self):
        """
        Number of items in the dataset.
        """
        return self._itemCount()

    def reset(self):
        """
        Restart data from index 0.
        """
        self.buffer_id = 0
        self._reset()

    def next(self):
        """
        return one minibatch in a (data, targets) tuple
        """
        (data, targets) = self._next(self.buffer_id)

        # Toggle buffer_id between 0 and 1
        self.buffer_id = 1 - self.buffer_id

        return (data, targets)

    def __iter__(self):
        """
        never ending iterator over dataset.
        """
        while True:
            yield self.next()
