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


logger = logging.getLogger(__name__)


BufferPair = (ct.c_void_p) * 2


class DeviceParams(ct.Structure):
    _fields_ = [('type', ct.c_int),
                ('id', ct.c_int),
                ('dataCount', ct.c_int),
                ('dataSize', ct.c_int),
                ('targetCount', ct.c_int),
                ('targetSize', ct.c_int),
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
        manifest_file (str, optional):
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
        onehot (boolean, optional):
            If the targets are categorical and have to be converted to a one-hot
            representation.
        nclasses (int, optional):
            Number of classes, if this dataset is intended for a classification
            problem.
        subset_percent (int, optional):
            Value between 0 and 100 indicating what percentage of the dataset
            partition to use.  Defaults to 100.
    """

    def __init__(self, set_name, cache_dir,
                 media_cfg_string, manifest_file,
                 device_type, device_id,
                 batch_size,
                 shuffle=False, reshuffle=False,
                 onehot=False, nclasses=None, subset_percent=100):
        if onehot is True and nclasses is None:
            raise ValueError('nclasses must be specified for one-hot labels')

        self.set_name = set_name
        cache_dir = os.path.expandvars(os.path.expanduser(cache_dir))
        if not os.path.exists(cache_dir):
            raise IOError('Directory not found: %s' % cache_dir)
        self.macrobatchsize = 5000
        self.cache_dir = cache_dir
        parent_dir = os.path.split(cache_dir)[0]
        self.manifest_file = manifest_file

        self.media_cfg_string = media_cfg_string
        self.device_type, self.device_id, self.default_dtype = device_type, device_id, np.float32

        self.item_count = ct.c_int(0)
        self.batch_size = batch_size
        self.buffer_id = 0
        self.start_idx = 0

        self.shuffle = shuffle
        self.reshuffle = reshuffle

        self.onehot = onehot
        self.nclasses = nclasses
        self.subset_percent = int(subset_percent)
        self.load_library()
        self.alloc()
        self.start()
        atexit.register(self.stop)

    def load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, 'bin', 'loader.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        # self.loaderlib.test_printer.argtypes = [ct.c_char_p, ct.c_char_p]
        self.loaderlib.get_error_message.restype = ct.c_char_p
        self.loaderlib.start.restype = ct.c_void_p

        self.loaderlib.next.argtypes = [ct.c_void_p]
        self.loaderlib.stop.argtypes = [ct.c_void_p]
        self.loaderlib.reset.argtypes = [ct.c_void_p]



    def alloc(self):

        # def alloc_bufs(dim0, dtype):
        #     return [self.be.iobuf(dim0=dim0, dtype=dtype) for _ in range(2)]

        # def ct_cast(buffers, idx):
        #     return ct.cast(int(buffers[idx].raw()), ct.c_void_p)

        # def cast_bufs(buffers):
        #     return BufferPair(ct_cast(buffers, 0), ct_cast(buffers, 1))

        # self.data = alloc_bufs(self.datum_size, self.datum_dtype)
        # self.targets = alloc_bufs(self.target_size, self.target_dtype)
        # self.media_params.alloc(self)
        self.device_params = DeviceParams(self.device_type, self.device_id)
        # if self.datum_dtype == self.be.default_dtype:
        #     self.backend_data = None
        # else:
        #     self.backend_data = self.be.iobuf(self.datum_size, dtype=self.be.default_dtype)

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.batch_size)

    def start(self):
        """
        Launch background threads for loading the data.
        """

        # import pdb; pdb.set_trace()
        self.loader = self.loaderlib.start(
            ct.byref(self.item_count),
            ct.c_char_p(self.manifest_file),
            ct.c_char_p(self.cache_dir),
            ct.c_char_p(self.media_cfg_string),
            ct.POINTER(DeviceParams)(self.device_params),
            ct.c_int(self.batch_size),
            ct.c_int(self.subset_percent),
            ct.c_int(self.macrobatchsize),
            ct.c_int(0),
            ct.c_bool(self.shuffle),
            ct.c_bool(self.reshuffle)
            )
        self.ndata = self.item_count.value
        if self.loader is None:
            a = self.loaderlib.get_error_message()
            print a
            raise RuntimeError('Failed to start data loader.')

        import pdb; pdb.set_trace()

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
        end = min(start + self.batch_size, self.ndata)
        if end == self.ndata:
            self.start_idx = self.batch_size - (self.ndata - start)
        self.loaderlib.next(self.loader)

        if self.backend_data is None:
            data = self.data[self.buffer_id]
        else:
            # Convert data to the required precision.
            self.backend_data[:] = self.data[self.buffer_id]
            data = self.backend_data

        targets = self.targets[self.buffer_id]

        self.buffer_id = 1 if self.buffer_id == 0 else 0

        return (data, targets)

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.batch_size):
            yield self.next(start)

cfg_string = r"""{"media":"image",
                  "data_config":
                     {"height":        40,
                      "width":         40,
                      "channel_major": false,
                      "flip":          true},
                  "target_config":
                     {"binary": true}
                     }
                     """

dloader_args = dict(set_name="tag_test",
                    cache_dir="/scratch/alex/dloader_test",
                    media_cfg_string=cfg_string,
                    manifest_file="/scratch/alex/dloader_test/cifar_manifest.txt",
                    device_type=0, device_id=0, batch_size=128)
dd = DataLoader(**dloader_args)


for x, t in dd:
    import pdb; pdb.set_trace()
    print(x)

