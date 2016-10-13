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
import sys


class LoaderRuntimeError(RuntimeError):
    pass


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
        self._config = config

        self._buffer_id = 0
        self._item_index = 0

        self._load_library()

        self.minibatch_size = config['minibatch_size']

        # Use the backend's rng seed if it exists and no specified seed
        # otherwise stick with what has been specified in the config
        if getattr(backend, 'rng_seed') is not None and config.get('random_seed') is None:
            config['random_seed'] = backend.rng_seed

        # Launch background threads
        self.loader = self._start(json.dumps(config), backend)

        atexit.register(self._stop)

        # compute the number of minibatches which will be in the first epoch
        self._compute_nbatches()

    def _load_library(self):
        if (sys.version_info > (3, 0)):
            # Python 3 builds extensions with names like
            # aeon_lib.cpython-35m-x86_64-linux-gnu.so
            # So we use this to do the name resolution
            import importlib.util
            libpath = importlib.util.find_spec('aeon_lib').origin
        else:
            path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            libpath = os.path.join(path, 'aeon_lib.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.get_error_message.restype = ct.c_char_p
        self.loaderlib.start.restype = ct.c_void_p

        self.loaderlib.next.argtypes = [ct.c_void_p, ct.c_int]
        self.loaderlib.next.restype = ct.py_object

        self.loaderlib.shapes.argtypes = [ct.c_void_p]
        self.loaderlib.shapes.restype = ct.py_object

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
        raise LoaderRuntimeError(
            'error in loader: {}'.format(
                self.loaderlib.get_error_message()
            )
        )

    def _start(self, config, backend):
        """
        C api wrapper with exception handling
        """
        if not hasattr(backend, 'consume'):
            raise TypeError('backend must have a callable consume attr')

        loader = self.loaderlib.start(
            ct.c_char_p(config.encode(encoding='utf-8')),
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

    def _next(self, _buffer_id):
        """
        C api wrapper with exception handling
        """
        tup = self.loaderlib.next(
            self.loader, ct.c_int(_buffer_id)
        )

        if tup is None:
            self._raise_loader_error()

        return tup

    def shapes(self):
        """
        C api wrapper with exception handling
        """
        ret = self.loaderlib.shapes(self.loader)

        if ret is None:
            self._raise_loader_error()

        return ret

    def _reset(self):
        """
        C api wrapper with exception handling
        """
        if self.loaderlib.reset(self.loader) == -1:
            self._raise_loader_error()

    @property
    def item_count(self):
        """
        Number of items in the dataset.
        """
        return self._itemCount()

    @property
    def config(self):
        """
        Dataloader configuration
        """
        return self._config

    def reset(self):
        """
        Restart data from index 0.
        """
        self._buffer_id = 0
        self._item_index = 0

        self._reset()

    def next(self):
        """
        return one minibatch in a (data, targets) tuple
        """
        dtuple = self._next(self._buffer_id)

        # Toggle _buffer_id between 0 and 1
        self._buffer_id = 1 - self._buffer_id

        return dtuple

    def unending_iter(self):
        """
        never ending iterator over dataset.
        """
        while True:
            yield self.next()

    def _compute_nbatches(self):
        """
        compute the number of batches for the next epoch
        """
        self._nbatches = -((self._item_index - self.item_count) // self.minibatch_size)

    @property
    def ndata(self):
        return self.item_count

    @property
    def shape(self):
        return self.shapes()[0]

    @property
    def nbatches(self):
        """
        Returns the number of minibatches in this dataset in this epoch.

        Sometimes the number of minibatches per epoch changes since items from
        the next epoch will be used to fill the last minibatch of an epoch if
        it doesn't line up with the end of the epoch.
        """
        return self._nbatches

    def __iter__(self):
        """
        Iterate over a single epoch.

        If the last minibatch is not full, use data from the next epoch to fill
        it.  The next epoch will not repeat the data used to fill the last
        minibatch.
        """
        for _ in range(self._nbatches):
            try:
                yield self.next()
            except LoaderRuntimeError as e:
                # TODO: log this somewhere instead of printing
                print(e)
            finally:
                # keep track of where we are in the dataset so we know which epoch we are on
                self._item_index += self.minibatch_size
                if self._item_index >= self.item_count:
                    self._item_index -= self.item_count
                    self._compute_nbatches()

    # UNUSED
    # these are reference for how we could switch to explicit epoch handling
    # rather than the implicit handling we have right now.

    def start_item_index(self, epoch_index):
        """ The starting item offset for epoch # `epoch_index`

        assumes 0 indexed epochs """
        return (self.item_count * epoch_index) % self.minibatch_size

    def nbatches_at_epoch(self, epoch_index):
        """ returns the number of minibatches which will be in epoch #
        `epoch_index` """
        return -((self.start_item_index(epoch_index) - self.item_count) // self.minibatch_size)

    def minibatch_index(self, epoch_index):
        """ returns the minibatch_index that `epoch_index` starts with """
        return (self.item_count * epoch_index) // self.minibatch_size
