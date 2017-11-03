#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Intel(R) Nervana(TM)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os
import sys
import getopt
from aeon import DataLoader

def get_help():
    return 'remote_iterator.py -a <address> -p <port> -m <manifest> -r <rdma_address> -s <rdma_port>'

def parse_input():
    argv = sys.argv[1:]
    address = ''
    port = ''
    manifest = ''
    rdma_address=''
    rdma_port=''
    try:
        opts, args = getopt.getopt(argv, "ha:p:m:r:s:", ["address=", "port=", "manifest=", "rdma_address=", "rdma_port="])
    except getopt.GetoptError:
        print(get_help())
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print(get_help())
            sys.exit()
        elif opt in ("-a", "--address"):
            address = arg
        elif opt in ("-p", "--port"):
            port = arg
        elif opt in ("-m", "--manifest"):
            manifest = arg
        elif opt in ("-r", "--rdma_address"):
            rdma_address = arg
        elif opt in ("-s", "--rdma_port"):
            rdma_port = arg
    if address == "":
        sys.exit('address parameter is required. Try --help for more information.')
    if port == "":
        sys.exit('port parameter is required. Try --help for more information.')
    if manifest == "":
        sys.exit('manifest parameter is required. Try --help for more information.')
    if (not rdma_address) != (not rdma_port):
        sys.exit('both rdma_address and rdma_port need to be set')
    return address, port, manifest, rdma_address, rdma_port


def main():
    address, port, manifest, rdma_address, rdma_port = parse_input()
    cache_root = "" # don't create cache
    batch_size = 4

    cfg = {
               'manifest_filename': manifest,
               'manifest_root': os.path.dirname(manifest),
               'batch_size': batch_size,
               'cache_directory': cache_root,
               'etl': [
                   {'type': 'image',
                    'width': 28,
                    'height': 28,
                    'channels': 1},
                   {'type': 'label',
                    'binary': False}
               ],
               'remote': {'address': address, 'port': int(port)}
           }

    # Add RDMA parameters if they are set
    if rdma_address:
        cfg['remote']['rdma_address'] = rdma_address
        cfg['remote']['rdma_port'] = int(rdma_port)

    # Create new aeon DataLoader object
    loader = DataLoader(config=cfg)
    print("data size: {0}".format(len(loader)))

    # Retrieve shapes
    shapes = loader.axes_info
    print("shapes: {0}".format(shapes))

    # Iterate through all available batches
    for batch in loader:
        image = batch[0]
        label = batch[1]

        print("{0} data: {1}".format(image[0], image[1]))
        print("{0} data: {1}".format(label[0], label[1]))

if __name__ == "__main__":
    main()
