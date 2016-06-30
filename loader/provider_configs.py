import json

def make_cifar_config(minibatch_size):
    """
    This is the configuration for doing random crops on cifar 10
    """
    dcfg = dict(height=40, width=40, channel_major=False, flip=True)
    tcfg = dict(binary=True)

    cfg_dict = dict(media="image",
                    data_config=dcfg,
                    target_config=tcfg,
                    manifest_filename="/scratch/alex/dloader_test/cifar_manifest.txt",
                    cache_directory="/scratch/alex/dloader_test",
                    macrobatch_size=5000, minibatch_size=minibatch_size)

    cifar_cfg_string = json.dumps(cfg_dict)
    return cifar_cfg_string
