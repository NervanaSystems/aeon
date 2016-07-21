import json
import os

def make_cifar_config(minibatch_size):
    """
    This is the configuration for doing random crops on cifar 10
    """
    dcfg = dict(type="image", config=dict(height=40, width=40, channel_major=False, flip=True))
    tcfg = dict(type="label", config=dict(binary=True))

    cfg_dict = dict(media="image_label",
                    data_config=dcfg,
                    target_config=tcfg,
                    manifest_filename="/scratch/alex/dloader_test/cifar_manifest_shuffle.txt",
                    cache_directory="/scratch/alex/dloader_test",
                    macrobatch_size=minibatch_size,
                    minibatch_size=minibatch_size)
    return cfg_dict


def make_miniplaces_config(manifest_dir="/scratch/alex/places2mini", minibatch_size=128):
    dcfg = dict(type="image", config=dict(height=112, width=112, channel_major=True, flip=True))
    tcfg = dict(type="label", config=dict(binary=False))
    macrobatch_size = 5000

    cfg_dict = dict(media="image_label",
                    data_config=dcfg,
                    target_config=tcfg,
                    manifest_filename=os.path.join(manifest_dir, "train.csv"),
                    cache_directory=os.path.join(manifest_dir, "cpio_cache"),
                    macrobatch_size=macrobatch_size,
                    minibatch_size=minibatch_size)

    return cfg_dict

def make_cstr_config(manifest_dir="/mnt/data/tyler/raw/VCTK-Corpus",
                     minibatch_size=128):
    dcfg = dict(type="audio", config=dict(sampling_freq=16000,
                                          clip_duration="3 seconds",
                                          frame_duration=""))
    tcfg = dict(type="transcript", config=dict(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()",
                                               max_length=25))

    cfg_dict = dict(media="audio_transcript",
                    data_config=dcfg,
                    target_config=tcfg,
                    manifest_filename=os.path.join(manifest_dir,
                                                   "vctk_manifest.csv"),
                    macrobatch_size=minibatch_size,
                    minibatch_size=minibatch_size)

    return cfg_dict
