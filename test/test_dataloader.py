import tempfile

import numpy as np
from PIL import Image as PILImage
import random
import struct
import pytest
import json
import os
import math

from aeon import DataLoader
from mock_data import random_manifest, generic_config, invalid_image

batch_size = 2

def test_loader_invalid_config_type():
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)

    cfg = json.loads(config)
    cfg["etl"][0]["type"] = 'invalid type name'
    config = json.dumps(cfg)

    with pytest.raises(RuntimeError) as ex:
        dl = DataLoader(config)
    assert 'unsupported' in str(ex)



def test_loader_missing_config_field():
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)

    cfg = json.loads(config)
    del cfg['etl'][0]["height"]
    config = json.dumps(cfg)

    with pytest.raises(RuntimeError) as ex:
        dl = DataLoader(config)
    assert 'height' in str(ex)


def test_loader_non_existant_manifest():
    config = generic_config('/this_manifest_file_does_not_exist', batch_size)

    with pytest.raises(RuntimeError) as ex:
        dl = DataLoader(config)
    assert "doesn't exist" in str(ex)


def test_loader_invalid_manifest():
    filename = tempfile.mkstemp()[1]
    config = generic_config(invalid_image(filename), batch_size)

    with pytest.raises(Exception) as ex:
        dl = DataLoader(config)
    assert 'must be string, but is null' in str(ex)


def test_loader():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    for i in range(1, 10):
        manifest = random_manifest(i)
        config = generic_config(manifest.name, batch_size)

        dl = DataLoader(config)

        assert len(list(iter(dl))) == math.ceil(float(i)/batch_size)


def test_loader_repeat_iter():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)

    dl = DataLoader(config)

    assert len(list(iter(dl))) == math.ceil(10./batch_size)


def test_loader_exception_next():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path+'/test_data')
    manifest = open("manifest.csv")

    config = generic_config(manifest.name, batch_size)
    dl = DataLoader(config)
    num_of_batches_in_manifest = 60
    for x in range(0, num_of_batches_in_manifest):
        dl.next()
    with pytest.raises(StopIteration) as ex:
        dl.next()
    manifest.close()
    os.chdir(cwd)


def test_loader_exception_iter():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path+'/test_data')
    manifest = open("manifest.csv")

    config = generic_config(manifest.name, batch_size)
    dl = DataLoader(config)

    num_of_manifest_entries = 120.
    assert len(list(iter(dl))) == math.ceil(num_of_manifest_entries/batch_size)

    manifest.close()
    os.chdir(cwd)


def test_loader_reset():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)
    dl = DataLoader(config)
    assert len(list(iter(dl))) == math.ceil(10./batch_size)
    dl.reset()
    assert len(list(iter(dl))) == math.ceil(10./batch_size)

if __name__ == '__main__':
    pytest.main()
