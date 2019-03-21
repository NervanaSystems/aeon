import tempfile

import numpy as np
from PIL import Image as PILImage
import random
import struct
import pytest
import os
import atexit

temp_files = []
first_time = True

def delete_temps():
    for f in temp_files:
        os.remove(f)

def random_image(filename):
    """
    generate a small random image
    """
    a = np.random.random((2, 2, 3)).astype('uint8')
    img = PILImage.fromarray(a)
    img.save(filename)


def invalid_image(filename):
    """
    write an empty file to filename to trigger invalid image file exceptions
    """
    with open(filename, 'w') as f:
        pass


def random_target(filename):
    target = int(random.random() * 1024)

    with open(filename, 'wb') as f:
        f.write(struct.pack('i', target))

    return filename


def random_manifest(num_lines, invalid_image_index=None):
    global first_time
    if first_time is True:
        first_time = False
        atexit.register(delete_temps)

    manifest = tempfile.NamedTemporaryFile(mode='w')
    manifest.write("@FILE\tFILE\n")
    # generate a manifest of filenames with an invalid image on the 3rd line
    for i in range(num_lines):
        img_filename = tempfile.mkstemp(suffix='.jpg')[1]
        if i == invalid_image_index:
            invalid_image(img_filename)
        else:
            random_image(img_filename)

        target_filename = tempfile.mkstemp(suffix='.txt')[1]
        random_target(target_filename)
        temp_files.append(img_filename)
        temp_files.append(target_filename)

        manifest.write("{}\t{}\n".format(img_filename, target_filename))
        with open(target_filename, 'w') as t:
            t.write(str(random.randint(0, 3)))
    manifest.flush()

    return manifest


def generic_config(manifest_name, batch_size):
    return {"manifest_filename": manifest_name,
        "batch_size": batch_size,
        "etl": [{"type": "image","height": 2,"width": 2}, {"type": "label", "binary": False}]
        }
