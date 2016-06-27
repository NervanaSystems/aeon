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
        return

    def consume(self, hostlist, devlist, buf_index):
        assert(buf_index < 2, "Can only double buffer")
        if devlist[buf_index] is None:



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
        self.load_library()
        self.start()
        atexit.register(self.stop)

    def load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, 'bin', 'loader.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.get_error_message.restype = ct.c_char_p
        self.loaderlib.start.restype = ct.c_void_p

        self.loaderlib.next.argtypes = [ct.c_void_p]
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
                ct.c_int(self.batch_size),
                ct.c_char_p(self.loader_cfg_string),
                ct.POINTER(DeviceParams)(self.device_params)
                )

        self.ndata = self.item_count.value
        if self.loader is None:
            a = self.loaderlib.get_error_message()
            raise RuntimeError('Failed to start data loader.' + a)

        self.data = self.attach_typeinfo(self.device_params.dtmInfo, self.device_params.data)
        self.targets = self.attach_typeinfo(self.device_params.tgtInfo, self.device_params.targets)

        import pdb; pdb.set_trace()

    def attach_typeinfo(self, typeinfo, pointers):
        """
        Takes the data and target buffers allocated by loaderlib and wraps them as numpy
        tensors with the appropriate datatypes
        """

        buf_interface = ct.pythonapi.PyBuffer_FromMemory
        buf_interface.restype = ct.py_object

        typename = np.dtype('{}{}'.format(typeinfo.Type[0], typeinfo.Size)).name
        ctdtype = ct.POINTER(getattr(ct, 'c_{}'.format(typename)))
        npdtype = getattr(np, typename)
        bufsize = typeinfo.Count * typeinfo.Size * self.batch_size
        shape = (typeinfo.Count, self.batch_size)

        res = []
        for dptr in pointers:
            b = buf_interface(ct.cast(dptr, ctdtype), bufsize)
            res.append(np.frombuffer(b, dtype=npdtype).reshape(shape))
        return res

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
        import pdb; pdb.set_trace()

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

# This is the configuration for doing random crops on cifar 10
dcfg = dict(height=40, width=40, channel_major=False, flip=True)
tcfg = dict(binary=True)

cfg_dict = dict(media="image",
                data_config=dcfg,
                target_config=tcfg,
                manifest_filename="/scratch/alex/dloader_test/cifar_manifest.txt",
                cache_directory="/scratch/alex/dloader_test",
                macrobatch_size=5000)



cfg_string = json.dumps(cfg_dict)

dloader_args = dict(set_name="tag_test",
                    loader_cfg_string=cfg_string,
                    device_type=0, device_id=0, batch_size=128)
dd = DataLoader(**dloader_args)


for x, t in dd:
    import pdb; pdb.set_trace()
    print(x)

