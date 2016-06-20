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
import logging
import numpy as np
import os
import atexit

from .media import MediaParams

logger = logging.getLogger(__name__)


BufferPair = (ct.c_void_p) * 2


class DeviceParams(ct.Structure):
    _fields_ = [('type', ct.c_int),
                ('id', ct.c_int),
                ('data', BufferPair),
                ('targets', BufferPair)]


class DataLoader(object):
    """
    Encapsulates the data loader library and exposes an API to iterate over
    generic data (images, video or audio given in compressed form). An index
    file that maps the data examples to their targets is expected to be provided
    in CSV format.

    Arguments:
        set_name (str):
            Name of this dataset partition.  This is used as prefix for
            directories and index files that may be created while ingesting.
        cache_dir (str):
            Directory to find the data.  This may also be used as the output
            directory to store ingested data.
        media_params (MediaParams):
            Parameters specific to the media type of the input data.
        target_size (int):
            The size of the targets.  For example: if the target is a class
            label, set this parameter to 1, indicating a single integer.  If
            the target is a mask image, the number of pixels in that image
            should be specified.
        target_conversion (str, optional):
            Specifies the method to be used for converting the targets that are
            provided in the index file.  The options are "no_conversion",
            "ascii_to_binary", "char_to_index" and "read_contents".  If this
            parameter is set to "read_contents", the targets given in the index
            file are treated as pathnames and their contents read in.  Defaults
            to "ascii_to_binary".
        index_file (str, optional):
            CSV formatted index file that defines the mapping between each
            example and its target.  The first line in the index file is
            assumed to be a header and is ignored.  Two columns are expected in
            the index.  The first column should be the file system path to
            individual data examples.  The second column may contain the actual
            label or the pathname of a file that contains the labels (e.g. a
            mask image).  If this parameter is not specified, creation of an
            index file is attempted.  Automaitic index generation can only be
            performed if the dataset is organized into subdirectories, which
            also represent labels.
        shuffle (boolean, optional):
            Whether to shuffle the order of data examples as the data is
            ingested.
        reshuffle (boolean, optional):
            Whether to reshuffle the order of data examples as they are loaded.
            If this is set to True, the order is reshuffled for each epoch.
            Useful for batch normalization.  Defaults to False.
        datum_type (data-type, optional):
            Data type of input data.  Defaults to np.uint8.
        target_type (data-type, optional):
            Data type of targets.  Defaults to np.int32.
        onehot (boolean, optional):
            If the targets are categorical and have to be converted to a one-hot
            representation.
        nclasses (int, optional):
            Number of classes, if this dataset is intended for a classification
            problem.
        subset_percent (int, optional):
            Value between 0 and 100 indicating what percentage of the dataset
            partition to use.  Defaults to 100.
        alphabet (str, optional):
            Alphabet to use for converting string labels.  This is only
            applicable if target_conversion is set to "char_to_index".
    """

    def __init__(self, set_name, cache_dir,
                 media_cfg_string, target_size,
                 index_file,
                 shuffle=False, reshuffle=False,
                 datum_dtype=np.uint8, target_dtype=np.int32,
                 onehot=True, nclasses=None, subset_percent=100):
        if onehot is True and nclasses is None:
            raise ValueError('nclasses must be specified for one-hot labels')

        self.set_name = set_name
        cache_dir = os.path.expandvars(os.path.expanduser(cache_dir))
        if not os.path.exists(cache_dir):
            raise IOError('Directory not found: %s' % cache_dir)
        self.macrobatchsize = 5000
        self.cache_dir = cache_dir
        parent_dir = os.path.split(cache_dir)[0]
        self.archive_dir = os.path.join(parent_dir, set_name + '-ingested')
        self.item_count = ct.c_int(0)
        self.bsz = self.be.bsz
        self.buffer_id = 0
        self.start_idx = 0
        self.media_params = media_params
        self.datum_size = media_params.datum_size()
        self.target_size = target_size

        self.shuffle = shuffle
        self.reshuffle = reshuffle
        self.datum_dtype = datum_dtype
        self.target_dtype = target_dtype
        self.onehot = onehot
        self.nclasses = nclasses
        self.subset_percent = int(subset_percent)
        self.ingest_params = ingest_params
        self.load_library()
        self.alloc()
        self.start()
        atexit.register(self.stop)

    def load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, os.pardir, os.pardir, 'loader', 'bin', 'loader.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.start.restype = ct.c_void_p
        self.loaderlib.next.argtypes = [ct.c_void_p]
        self.loaderlib.stop.argtypes = [ct.c_void_p]
        self.loaderlib.reset.argtypes = [ct.c_void_p]

    def alloc(self):

        def alloc_bufs(dim0, dtype):
            return [self.be.iobuf(dim0=dim0, dtype=dtype) for _ in range(2)]

        def ct_cast(buffers, idx):
            return ct.cast(int(buffers[idx].raw()), ct.c_void_p)

        def cast_bufs(buffers):
            return BufferPair(ct_cast(buffers, 0), ct_cast(buffers, 1))

        # self.data = alloc_bufs(self.datum_size, self.datum_dtype)
        # self.targets = alloc_bufs(self.target_size, self.target_dtype)
        self.media_params.alloc(self)
        self.device_params = DeviceParams(self.be.device_type, self.be.device_id)
        if self.onehot:
            self.onehot_labels = self.be.iobuf(self.nclasses, dtype=self.be.default_dtype)

        if self.datum_dtype == self.be.default_dtype:
            self.backend_data = None
        else:
            self.backend_data = self.be.iobuf(self.datum_size, dtype=self.be.default_dtype)

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.bsz)

    def start(self):
        """
        Launch background threads for loading the data.
        """
        self.loader = self.loaderlib.start(
            ct.byref(self.item_count), ct.c_int(self.bsz),
            self.shuffle, self.reshuffle,
            ct.c_int(self.subset_percent),
            ct.POINTER(DeviceParams)(self.device_params),
            ct.c_char_p(self.manifest_file.encode()),
            ct.c_int(self.macrobatchsize),
            ct.c_char_p(self.cache_dir.encode()),
            ct.c_int(0))
        self.ndata = self.item_count.value
        if self.loader is None:
            raise RuntimeError('Failed to start data loader.')

    def stop(self):
        """
        Clean up and exit background threads.
        """
        self.loaderlib.stop(self.loader)

    def reset(self):
        """
        Restart data from index 0
        """
        self.buffer_id = 0
        self.start_idx = 0
        self.loaderlib.reset(self.loader)

    def next(self, start):
        end = min(start + self.bsz, self.ndata)
        if end == self.ndata:
            self.start_idx = self.bsz - (self.ndata - start)
        self.loaderlib.next(self.loader)

        if self.backend_data is None:
            data = self.data[self.buffer_id]
        else:
            # Convert data to the required precision.
            self.backend_data[:] = self.data[self.buffer_id]
            data = self.backend_data

        if self.onehot:
            # Convert labels to one-hot encoding.
            self.onehot_labels[:] = self.be.onehot(self.targets[self.buffer_id], axis=0)
            targets = self.onehot_labels
        else:
            targets = self.targets[self.buffer_id]

        meta = self.meta[self.buffer_id]
        self.buffer_id = 1 if self.buffer_id == 0 else 0
        return self.media_params.process(self, data, targets, meta)

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            yield self.next(start)
