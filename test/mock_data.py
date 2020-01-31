import tempfile

import numpy as np
from PIL import Image as PILImage
import random
import struct
import os
import atexit

TEMP_FILES = []
FIRST_TIME = True

def delete_temps():
    for f in TEMP_FILES:
        os.remove(f)

def random_image(filename):
    """
    generate a small random image
    """
    a = np.random.random((2, 2, 3)).astype('uint8')
    img = PILImage.fromarray(a)
    img.save(filename)
    os.chmod(filename, 0o600)


def invalid_image(filename):
    """
    write an empty file to filename to trigger invalid image file exceptions
    """
    with open(filename, 'w') as f:
        os.chmod(filename, 0o600)
        pass

def broken_image(filename):
    """
    generate a small random broken image
    """
    random_image(filename)
    os.chmod(filename, 0o600)
    with open(filename, 'r+b') as file:
        file.seek(3)
        file.write(bytearray(b'{\x00\x00d'))


def random_target(filename):
    target = int(random.random() * 1024)

    with open(filename, 'wb') as f:
        os.chmod(filename, 0o600)
        f.write(struct.pack('i', target))

    return filename


def random_manifest(num_lines, invalid_image_index=None, broken_image_index=None):
    assert broken_image_index is None or broken_image_index != invalid_image_index
    global FIRST_TIME, TEMP_FILES
    if FIRST_TIME is True:
        FIRST_TIME = False
        atexit.register(delete_temps)

    manifest = tempfile.NamedTemporaryFile(mode='w')
    manifest.write("@FILE\tFILE\n")
    # generate a manifest of filenames with broken and invalid images at specified indexes
    for i in range(num_lines):
        fid, img_filename = tempfile.mkstemp(suffix='.jpg')
        if i == invalid_image_index:
            invalid_image(img_filename)
        elif i == broken_image_index:
            broken_image(img_filename)
        else:
            random_image(img_filename)
        os.close(fid)

        fid, target_filename = tempfile.mkstemp(suffix='.txt')
        random_target(target_filename)
        os.close(fid)

        TEMP_FILES.append(img_filename)
        TEMP_FILES.append(target_filename)

        manifest.write("{}\t{}\n".format(img_filename, target_filename))
        with open(target_filename, 'w') as t:
            os.chmod(target_filename, 0o600)
            t.write(str(random.randint(0, 3)))
    manifest.flush()

    return manifest


def generic_config(manifest_name, batch_size):
    return {"manifest_filename": manifest_name,
        "cpu_list": "0",
        "batch_size": batch_size,
        "etl": [{"type": "image","height": 2,"width": 2}, {"type": "label", "binary": False}]
        }
