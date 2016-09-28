import tempfile

import numpy as np
from PIL import Image as PILImage
import random
import struct
import pytest

from aeon import DataLoader, LoaderRuntimeError

from neon.backends import gen_backend
from neon.models import Model


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
    manifest = tempfile.NamedTemporaryFile(mode='w')

    # generate a manifest of filenames with an invalid image on the 3rd line
    for i in range(num_lines):
        img_filename = tempfile.mkstemp(suffix='.jpg')[1]
        if i == invalid_image_index:
            invalid_image(img_filename)
        else:
            random_image(img_filename)

        target_filename = tempfile.mkstemp(suffix='.jpg')[1]
        random_target(target_filename)

        manifest.write("{},{}\n".format(img_filename, target_filename))
    manifest.flush()

    return manifest


def generic_config(manifest_name):
    return {
        'manifest_filename': manifest_name,
        'minibatch_size': 2,
        'image': {
            'height': 2,
            'width': 2,
        },
        'label': {'binary': True},
        'type': 'image,label',
    }


def test_loader_invalid_config_type():
    manifest = random_manifest(10)
    config = generic_config(manifest.name)

    config['type'] = 'invalid type name'

    with pytest.raises(Exception) as ex:
        dl = DataLoader(config, gen_backend(backend='cpu'))

    #assert 'is not supported' in str(ex)


def test_loader_missing_config_field():
    manifest = random_manifest(10)
    config = generic_config(manifest.name)

    del config['image']

    with pytest.raises(Exception) as ex:
        dl = DataLoader(config, gen_backend(backend='cpu'))

    assert 'image' in str(ex)


def test_loader_non_existant_manifest():
    config = generic_config('/this_manifest_file_does_not_exist')

    with pytest.raises(Exception):
        dl = DataLoader(config, gen_backend(backend='cpu'))


def test_loader_invalid_manifest():
    filename = tempfile.mkstemp()[1]
    config = generic_config(invalid_image(filename))

    with pytest.raises(Exception):
        dl = DataLoader(config, gen_backend(backend='cpu'))


def test_loader():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name)

    dl = DataLoader(config, gen_backend(backend='cpu'))

    assert len(list(iter(dl))) == 5


def test_loader_repeat_iter():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name)

    dl = DataLoader(config, gen_backend(backend='cpu'))

    assert len(list(iter(dl))) == 5
    assert len(list(iter(dl))) == 5


def test_loader_exception_next():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10, 2)
    config = generic_config(manifest.name)

    dl = DataLoader(config, gen_backend(backend='cpu'))
    dl.next()
    with pytest.raises(LoaderRuntimeError):
        dl.next()


def test_loader_exception_iter():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10, 2)
    config = generic_config(manifest.name)

    dl = DataLoader(config, gen_backend(backend='cpu'))

    assert len(list(iter(dl))) == 4


def test_loader_reset():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name)

    dl = DataLoader(config, gen_backend(backend='cpu'))

    assert len(list(iter(dl))) == 5
    dl.reset()
    assert len(list(iter(dl))) == 5


if __name__ == '__main__':
    pytest.main()
