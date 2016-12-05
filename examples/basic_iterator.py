from aeon import AeonDataloader

manifest_file = "loader/test/test_data/manifest.csv"
batch_size = 128
cache_root = ""

config = {
           'manifest_filename': manifest_file,
           'minibatch_size': batch_size,
           'macrobatch_size': 25000,
           'cache_directory': cache_root,
           'type': 'image,label',
           'image': {'height': 28,
                     'width': 28,
                     'channels': 1},
           'label': {'binary': False}
        }

d1 = AeonDataloader(config)
names = d1.get_buffer_names()
print("names {0}").format(names)
d1.get_buffer_shape("test_name")

print("d1 length {0}").format(len(d1))

shapes = d1.shapes()

for x in d1:
    print("d1 {0}").format(x)

d1.reset()

for x in d1:
    print("d1 {0}").format(x)
