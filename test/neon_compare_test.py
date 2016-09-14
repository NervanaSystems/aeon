# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
"""
Verify that different ways of loading datasets lead to the same result.

This test utility accepts the same command line parameters as neon. It
downloads the CIFAR-10 dataset and saves it as individual PNG files. It then
proceeds to fit and evaluate a model using two different ways of loading the
data.

run as follows:
python compare.py -e1 -r0 -w <place where data lives>

"""
import numpy as np
import os
from neon import NervanaObject
from neon.data import ArrayIterator
from neon.initializers import Uniform
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, Rectlin, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.util.persist import get_data_cache_dir, ensure_dirs_exist
from neon.data import CIFAR10, AeonDataLoader
from neon.data.dataloader_transformers import OneHot, TypeCast, BGRMeanSubtract
from PIL import Image
from tqdm import tqdm


bgr_means = [127, 119, 104]


def ingest_cifar10(out_dir, overwrite=False):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    dataset = dict()
    cifar10 = CIFAR10(path=out_dir, normalize=False)
    dataset['train'], dataset['val'], _ = cifar10.load_data()

    set_names = ('train', 'val')
    manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        return manifest_files

    # Write out label files and setup directory structure
    lbl_paths, img_paths = dict(), dict(train=dict(), val=dict())
    for lbl in range(10):
        lbl_paths[lbl] = ensure_dirs_exist(os.path.join(out_dir, 'labels', str(lbl) + '.txt'))
        np.savetxt(lbl_paths[lbl], [lbl], fmt='%d')
        for setn in ('train', 'val'):
            img_paths[setn][lbl] = ensure_dirs_exist(os.path.join(out_dir, setn, str(lbl) + '/'))

    np.random.seed(0)
    # Now write out image files and manifests
    for setn, manifest in zip(set_names, manifest_files):
        records = []
        for idx, (img, lbl) in tqdm(enumerate(zip(*dataset[setn]))):
            img_path = os.path.join(img_paths[setn][lbl[0]], str(idx) + '.png')
            im = img.reshape((3, 32, 32))
            im = Image.fromarray(np.uint8(np.transpose(im, axes=[1, 2, 0]).copy()))
            im.save(img_path, format='PNG')
            records.append((img_path, lbl_paths[lbl[0]]))

        np.random.shuffle(records)
        np.savetxt(manifest, records, fmt='%s,%s')

    return manifest_files


def make_aeon_config(manifest_filename, minibatch_size):
    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=5000,
        type='image,label',
        label={'binary': False},
        image={'height': 32, 'width': 32})


def transformers(dl):
    dl = OneHot(dl, nclasses=10, index=1)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = BGRMeanSubtract(dl, index=0, pixel_mean=bgr_means)
    return dl


def load_dataset(manifest):
    with open(manifest) as fd:
        lines = fd.readlines()
    assert len(lines) > 0, 'could not read %s' % manifest

    data = None
    for idx, line in enumerate(lines):
        imgfile, labelfile = line.split(',')
        labelfile = labelfile[:-1]
        # Convert from RGB to BGR to be consistent with the data loader
        im = np.asarray(Image.open(imgfile))[:, :, ::-1]
        # Convert from HWC to CHW
        im = np.transpose(im, axes=[2, 0, 1]).ravel()
        if data is None:
            data = np.empty((len(lines), im.shape[0]), dtype='float32')
            labels = np.empty((len(lines), 1), dtype='int32')
        data[idx] = im
        with open(labelfile) as fd:
            labels[idx] = int(fd.read())
    data_view = data.reshape((data.shape[0], 3, -1))
    # Subtract mean values of B, G, R
    data_view -= np.array(bgr_means).reshape((1, 3, 1))
    return (data, labels)


def load_cifar10_imgs(train_manifest, val_manifest):
    (X_train, y_train) = load_dataset(train_manifest)
    (X_test, y_test) = load_dataset(val_manifest)
    return (X_train, y_train), (X_test, y_test), 10


def run(args, train, test):
    init_uni = Uniform(low=-0.1, high=0.1)
    opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                      momentum_coef=0.9,
                                      stochastic_round=args.rounding)
    layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=True),
              Pooling((2, 2)),
              Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=True),
              Pooling((2, 2)),
              Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=True),
              Affine(nout=10, init=init_uni, activation=Softmax())]
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    mlp = Model(layers=layers)
    callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)
    mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
    err = mlp.eval(test, metric=Misclassification())*100
    print('Misclassification error = %.2f%%' % err)
    return err


def test_iterator():
    print('Testing data iterator')
    NervanaObject.be.gen_rng(args.rng_seed)
    image_dir = get_data_cache_dir(args.data_dir, subdir='extracted')
    train_manifest, val_manifest = ingest_cifar10(out_dir=image_dir)

    (X_train, y_train), (X_test, y_test), nclass = load_cifar10_imgs(train_manifest, val_manifest)
    train = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))
    test = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))
    return run(args, train, test)


def test_loader():
    print('Testing data loader')
    NervanaObject.be.gen_rng(args.rng_seed)
    image_dir = get_data_cache_dir(args.data_dir, subdir='extracted')
    train_manifest, val_manifest = ingest_cifar10(out_dir=image_dir)

    train_config = make_aeon_config(train_manifest, args.batch_size)
    val_config = make_aeon_config(val_manifest, args.batch_size)

    train = transformers(AeonDataLoader(train_config, NervanaObject.be))
    test = transformers(AeonDataLoader(val_config, NervanaObject.be))

    err = run(args, train, test)
    return err

parser = NeonArgparser(__doc__)
args = parser.parse_args()
# Perform ingest if it hasn't already been done and return manifest files
assert test_loader() == test_iterator(), 'The results do not match'
print 'OK'
