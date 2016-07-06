from dloader import DataLoader
from protobackends import MultiGpuBackend
from provider_configs import make_cifar_config, make_miniplaces_config

batch_size = 128
mybackend = MultiGpuBackend(num_dev=1)
# myconfig = make_cifar_config(batch_size)
myconfig = make_miniplaces_config(minibatch_size=batch_size)

dd = DataLoader(backend=mybackend,
                loader_cfg_string=myconfig,
                batch_size=batch_size)

for i, (x, t) in enumerate(dd):
    host_x = mybackend.get_ary(x)
    host_t = mybackend.get_ary(t)

    print(i, host_x)
    print(i, host_t)
    if i == 10:
        break

