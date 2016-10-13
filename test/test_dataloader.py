import tempfile

import numpy as np
from PIL import Image as PILImage
import random
import struct
import pytest

from aeon import DataLoader, LoaderRuntimeError
from mock_data import random_manifest, generic_config, invalid_image

from neon.backends import gen_backend
from neon.models import Model


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
