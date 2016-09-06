import json
import os

def make_cifar_config(minibatch_size):
    """
    This is the configuration for doing random crops on cifar 10
    """
    dcfg = dict(height=40, width=40, channel_major=False, flip=True)
    tcfg = dict(binary=True)

    cfg_dict = dict(type="image,label",
                    image=dcfg,
                    label=tcfg,
                    manifest_filename="/scratch/alex/dloader_test/cifar_manifest_shuffle.txt",
                    cache_directory="/scratch/alex/dloader_test",
                    macrobatch_size=minibatch_size,
                    minibatch_size=minibatch_size)
    return cfg_dict


def make_miniplaces_config(manifest_dir="/scratch/alex/places2mini", minibatch_size=128):
    dcfg = dict(height=112, width=112, channel_major=True, flip=True)
    tcfg = dict(binary=True)
    macrobatch_size = 5000

    cfg_dict = dict(type="image,label",
                    image=dcfg,
                    label=tcfg,
                    manifest_filename=os.path.join(manifest_dir, "train.csv"),
                    cache_directory=os.path.join(manifest_dir, "cpio_cache"),
                    macrobatch_size=macrobatch_size,
                    minibatch_size=minibatch_size)

    return cfg_dict

def make_cstr_config(manifest_dir="/scratch/alex/audio/VCTK-Corpus/ingested",
                     minibatch_size=128):
    dcfg = dict(sampling_freq=16000,
                max_duration="3 seconds",
                frame_length="256 samples",
                frame_stride="128 samples",
                window_type="hann")

    tcfg = dict(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ-_!? .,()",
                max_length=25)

    cfg_dict = dict(type="audio,transcribe",
                    audio=dcfg,
                    transcript=tcfg,
                    manifest_filename=os.path.join(manifest_dir, "train.csv"),
                    cache_directory=os.path.join(manifest_dir, "cpio_cache"),
                    macrobatch_size=minibatch_size,
                    minibatch_size=minibatch_size)

    return cfg_dict
