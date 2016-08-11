.. ---------------------------------------------------------------------------
.. Copyright 2015 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Getting Started
===============

Installation
------------

First grab some prerequisites (at the very least)::

  sudo apt-get install libcurl4-openssl-dev clang

Then to install aeon::

  git clone https://github.com/NervanaSystems/aeon.git
  cd aeon
  sudo python setup.py install

Usage
-----
.. TODO: put a small, simpler example above this more complicated one.

This example is taken from the neon ``examples/alexnet.py`` repository. First, use the proper imports::
  
    from aeon import DataLoader

Define a configuration dictionary::

    def make_aeon_config(manifest_filename, minibatch_size, do_randomize=False, 
    subset_pct=100):
        image_decode_cfg = dict(
            height=224, width=224,
            scale=[0.875, 0.875],        # .875 fraction is 224/256 (short side)
            flip=do_randomize,           # whether to do random flips
            center=(not do_randomize))   # whether to do random crops

        return dict(
            manifest_filename=manifest_filename,
            minibatch_size=minibatch_size,
            macrobatch_size=1024,
            cache_dir=cpio_dir,
            subset_fraction=float(subset_pct/100.0),
            shuffle_manifest=do_randomize,
            shuffle_every_epoch=do_randomize,
            type='image,label',
            label={'binary': True},
            image=image_decode_cfg)


    train_config = make_aeon_config(os.path.join(manifest_dir, 'train_file.csv'),
                                    args.batch_size,
                                    do_randomize=True,
                                    subset_pct=args.subset_percent)

    valid_config = make_aeon_config(os.path.join(manifest_dir, 'val_file.csv'),
                                    args.batch_size)

And then we define a set of transformations that are applied sequentially to 
the DataLoader objects::

    def transformers(dl):
        dl = OneHot(dl, nclasses=1000, index=1)
        dl = TypeCast(dl, index=0, dtype=np.float32)
        dl = ImageMeanSubtract(dl, index=0, pixel_mean=[104.41227722, 119.21331787, 126.80609131])
        return dl

    train = transformers(DataLoader(train_config, model.be))
    valid = transformers(DataLoader(valid_config, model.be))


Finally, we use these configs for training the model (model definition not 
shown here)::

    # configure callbacks
    valmetric = TopKMisclassification(k=5)
    callbacks = Callbacks(model, eval_set=valid, metric=valmetric, **args.callback_args)
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

When running this example with ``python alexnet.py``, you should expect 
training to pause for ~10 seconds between runs of minibatches. This is expected 
and a result of the DataLoader dumping cached data into CPIO archives. Once 
this is complete for the entire dataset, then training will continue without IO 
induced pauses.
