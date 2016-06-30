from dloader import DataLoader
from protobackends import CpuBackend
from provider_configs import make_cifar_config

batch_size = 128
mybackend = CpuBackend()
myconfig = make_cifar_config(batch_size)

dd = DataLoader(backend=mybackend,
                loader_cfg_string=myconfig,
                batch_size=batch_size)

for i, (x, t) in enumerate(dd):
    print i, mybackend.get_ary(x)
    # import pdb; pdb.set_trace()
    if i == 10:
        break

