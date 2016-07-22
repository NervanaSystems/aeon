import os
from distutils.core import setup, Extension
import subprocess


def shell_stdout(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].strip()

for line in shell_stdout('bash -c "source env.sh; env"').split('\n'):
    key, _, value = line.partition("=")
    os.environ[key] = value

define_macros = []
include_dirs = os.environ['INC'].replace('-I', '').split()
library_dirs = os.environ['LDIR'].replace('-L', '').split()
libraries = os.environ['LIBS'].replace('-l', '').split()
extra_compile_args = os.environ['CFLAGS'].split()

module = Extension(
    'axon_lib',
    sources=[
        'loader/src/cpio.cpp',
        'loader/src/buffer_in.cpp',
        'loader/src/buffer_out.cpp',
        'loader/src/pyBackendWrapper.cpp',
        'loader/src/pyLoader.cpp',
        'loader/src/api_pversion.cpp',
        'loader/src/etl_audio.cpp',
        'loader/src/noise_clips.cpp',
        'loader/src/specgram.cpp',
        'loader/src/log.cpp',
        'loader/src/codec.cpp',
        'loader/src/image.cpp',
        'loader/src/etl_pixel_mask.cpp',
        'loader/src/etl_image.cpp',
        'loader/src/etl_image_var.cpp',
        'loader/src/etl_bbox.cpp',
        'loader/src/etl_label_map.cpp',
        'loader/src/etl_char_map.cpp',
        'loader/src/etl_localization.cpp',
        'loader/src/etl_video.cpp',
        'loader/src/util.cpp',
        'loader/src/wav_data.cpp',
        'loader/src/manifest.cpp',
        'loader/src/provider_factory.cpp',
        'loader/src/batch_file_loader.cpp',
        'loader/src/batch_loader_cpio_cache.cpp',
        'loader/src/sequential_batch_iterator.cpp',
        'loader/src/shuffled_batch_iterator.cpp',
        'loader/src/minibatch_iterator.cpp',
        'loader/src/batch_loader.cpp',
        'loader/src/box.cpp',
        'loader/src/buffer_pool.cpp',
        'loader/src/buffer_pool_in.cpp',
        'loader/src/buffer_pool_out.cpp',
    ],
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
)

setup(
    name='nervana-aeon',
    version='0.1',
    packages=['aeon'],
    ext_modules=[module],
)
