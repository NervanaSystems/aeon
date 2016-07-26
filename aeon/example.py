from dataloader import DataLoader
from protobackends import CpuBackend
from provider_configs import make_cifar_config, make_miniplaces_config, make_cstr_config

batch_size = 128
mybackend = CpuBackend()
# myconfig = make_cifar_config(batch_size)
myconfig = make_cstr_config()
# myconfig = make_miniplaces_config(minibatch_size=batch_size)

dd = DataLoader(backend=mybackend, config=myconfig)

for i, dtuple in enumerate(dd):
    host_x = mybackend.get_ary(dtuple[0])
    host_t = mybackend.get_ary(dtuple[1])

    print(i, host_x)
    print(i, host_t)
    if i == 10:
        break

