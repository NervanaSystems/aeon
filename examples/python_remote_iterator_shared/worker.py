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

import sys
import getopt
import time
from aeon import DataLoader

def get_help():
    return 'worker.py -a <address> -p <port> -i <session> -r <rdma_address> -s <rdma_port>'

def parse_input():
    argv = sys.argv[1:]
    address = session_id = ''
    session_id = ''
    rdma_address=''
    rdma_port=''
    try:
        opts, args = getopt.getopt(argv, "ha:p:i:r:s:", ["address=", "port=", "session_id=", "rdma_address=", "rdma_port="])
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
        elif opt in ("-i", "--session_id"):
            session_id = arg
        elif opt in ("-r", "--rdma_address"):
            rdma_address = arg
        elif opt in ("-s", "--rdma_port"):
            rdma_port = arg
    if address == "":
        sys.exit('address parameter is required. Try --help for more information.')
    if port == "":
        sys.exit('port parameter is required. Try --help for more information.')
    if session_id == "":
        sys.exit('session_id parameter is required. Try --help for more information.')
    if (not rdma_address) != (not rdma_port):
        sys.exit('both rdma_address and rdma_port need to be set')
    return address, port, session_id, rdma_address, rdma_port


def main():
    address, port, session_id, rdma_address, rdma_port = parse_input()
    cache_root = "" # don't create cache
    batch_size = 4

    cfg = {
            'remote': {'address': address, 'port': int(port), 'session_id': session_id, 'close_session': False}
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
    batch_counter = 1
    for batch in loader:
        print("Batch {0} ready.").format(batch_counter)
        batch_counter += 1
        time.sleep(1)

if __name__ == "__main__":
    main()

