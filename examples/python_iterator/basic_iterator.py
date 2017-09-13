#!/usr/bin/env python
import os
from aeon import DataLoader

pdir = os.path.dirname(os.path.abspath(__file__))
manifest_root = os.path.join(pdir, '..', '..', 'test', 'test_data')

manifest_file = os.path.join(manifest_root, 'manifest.csv')
cache_root = ""

cfg = {
           'manifest_filename': manifest_file,
           'manifest_root': manifest_root,
           'batch_size': 20,
           'block_size': 40,
           'cache_directory': cache_root,
           'etl': [
               {'type': 'image',
                'channel_major': False,
                'width': 28,
                'height': 28,
                'channels': 1},
               {'type': 'label',
                'binary': False}
           ]
        }

d1 = DataLoader(config=cfg)
print("d1 length {0}".format(len(d1)))

shapes = d1.axes_info
print("shapes: {0}".format(shapes))

for x in d1:
    image = x[0]
    label = x[1]

    print("{0} data: {1}".format(image[0], image[1]))
    print("{0} data: {1}".format(label[0], label[1]))
