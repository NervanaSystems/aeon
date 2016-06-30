from dloader import DataLoader
from protobackends import MultiGpuBackend
from provider_configs import make_cifar_config

batch_size = 128
mybackend = MultiGpuBackend()
myconfig = make_cifar_config(batch_size)

dd = DataLoader(backend=mybackend,
                loader_cfg_string=myconfig,
                batch_size=batch_size)

for i, (x, t) in enumerate(dd):
    host_x = mybackend.get_ary(x)

    print i, host_x
    if i == 10:
        break

