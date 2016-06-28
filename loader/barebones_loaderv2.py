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
import json
import pycuda

logger = logging.getLogger(__name__)

class ProtoBackend(object):
    def __init__(self):
        self.use_pinned_mem = True
        self.test_string = "Hey look at me"

    def consume(self, buf_index, hostlist, devlist):
        if buf_index >= 2:
            raise ValueError('Can only double buffer')
        print "test print", hostlist[0]
        if devlist[buf_index] is None:
            devlist[buf_index] = self.empty_like(hostlist[buf_index])
        print devlist[buf_index].shape, hostlist[buf_index.shape]
        devlist[buf_index][:] = hostlist[buf_index].T

    def empty_like(self, npary):
        return np.empty_like(npary.T)


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
        manifest_file (str, optional):
            CSV formatted index file that defines the mapping between each
            example and its target.  The first line in the index file is
            assumed to be a header and is ignored.  Two columns are expected in
            the index.  The first column should be the file system path to
            individual data examples.  The second column may contain the actual
            label or the pathname of a file that contains the labels (e.g. a
            mask image).  If this parameter is not specified, creation of an
            index file is attempted.  Automatic index generation can only be
            performed if the dataset is organized into subdirectories, which
            also represent labels.
        onehot (boolean, optional):
            If the targets are categorical and have to be converted to a one-hot
            representation.
        nclasses (int, optional):
            Number of classes, if this dataset is intended for a classification
            problem.
    """

    def __init__(self, set_name, batch_size,
                 loader_cfg_string,
                 onehot=False, nclasses=None):
        if onehot is True and nclasses is None:
            raise ValueError('nclasses must be specified for one-hot labels')

        self.set_name = set_name
        self.loader_cfg_string = loader_cfg_string
        self.batch_size = batch_size
        self.onehot = onehot
        self.nclasses = nclasses

        self.item_count = ct.c_int(0)
        self.buffer_id = 0
        self.start_idx = 0

        self.backend_data = None
        self.backend = ProtoBackend()
        self.load_library()
        self.start()
        atexit.register(self.stop)

    def load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, 'bin', 'loader.so')
        self.loaderlib = ct.PyDLL(libpath)
        # self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.get_error_message.restype = ct.c_char_p
        self.loaderlib.start.restype = ct.c_void_p

        self.loaderlib.next.argtypes = [ct.c_void_p, ct.c_int]
        self.loaderlib.next.restype = ct.py_object

        self.loaderlib.stop.argtypes = [ct.c_void_p]
        self.loaderlib.reset.argtypes = [ct.c_void_p]

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.batch_size)

    def start(self):
        """
        Launch background threads for loading the data.
        """
        self.loader = self.loaderlib.start(
                ct.byref(self.item_count),
                ct.c_char_p(self.loader_cfg_string),
                ct.py_object(self.backend)
                )

        self.ndata = self.item_count.value
        if self.loader is None:
            a = self.loaderlib.get_error_message()
            raise RuntimeError('Failed to start data loader.' + a)


        # import pdb; pdb.set_trace()


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


        (data, targets) = self.loaderlib.next(self.loader, ct.c_int(self.buffer_id))
        import pdb; pdb.set_trace()

        # (data, targets) = self.loaderlib.get_dtm_tgt(self.buffer_id)



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

# This is the configuration for doing random crops on cifar 10
dcfg = dict(height=40, width=40, channel_major=False, flip=True)
tcfg = dict(binary=True)

cfg_dict = dict(media="image",
                data_config=dcfg,
                target_config=tcfg,
                manifest_filename="/scratch/alex/dloader_test/cifar_manifest.txt",
                cache_directory="/scratch/alex/dloader_test",
                macrobatch_size=5000, minibatch_size=128)



cfg_string = json.dumps(cfg_dict)

dloader_args = dict(set_name="tag_test",
                    batch_size=cfg_dict['minibatch_size'],
                    loader_cfg_string=cfg_string)
# print threading.current_thread()
# print threading.enumerate()
dd = DataLoader(**dloader_args)


for x, t in dd:
    import pdb; pdb.set_trace()
    print(x)

