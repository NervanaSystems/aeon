import json

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

    cifar_cfg_string = json.dumps(cfg_dict)
    return cifar_cfg_string
