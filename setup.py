import os
from distutils.core import setup, Extension
import subprocess

os.environ['CC'] = 'clang++'
os.environ['CFLAGS'] = '-Wno-deprecated-declarations -std=c++11'


def shell_stdout(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].strip()

define_macros = []
include_dirs = []
library_dirs = []
libraries = []

if shell_stdout('pkg-config --exists opencv; echo $?') == '0':
    define_macros.append(('HAS_IMGLIB', '1'))
    include_dirs.extend(
        shell_stdout('pkg-config --cflags opencv').replace('-I', '').split()
    )
    library_dirs.extend(
        shell_stdout('pkg-config --libs-only-L opencv').replace('-L', '').split()
    )
    libraries.extend(
        shell_stdout('pkg-config --libs-only-l opencv').replace('-l', '').split()
    )

if shell_stdout('pkg-config --exists libavutil libavformat libavcodec libswscale; echo $?') == '0':
    define_macros.append(['HAS_VIDLIB', '1'])
    define_macros.append(['HAS_AUDLIB', '1'])

    library_dirs.extend(
        shell_stdout('pkg-config --libs-only-L libavutil libavformat libavcodec libswscale').replace('-L', '').split()
    )
    libraries.extend(['avutil', 'avformat', 'avcodec', 'swscale'])

if os.environ.get('HAS_GPU') == 'true':
    CUDA_ROOT = shell_stdout('which nvcc').replace('/bin/nvcc', '')
    if CUDA_ROOT:
        include_dirs.append('{CUDA_ROOT}/include'.format(CUDA_ROOT=CUDA_ROOT))

    define_macros.append(('HAS_GPU', '1'))
    include_dirs.append('{CUDA_ROOT}/include'.format(CUDA_ROOT=CUDA_ROOT))
    if shell_stdout('uname -s') == 'Darwin':
        library_dirs.append('{CUDA_ROOT}/lib'.format(CUDA_ROOT=CUDA_ROOT))
    else:
        library_dirs.append('{CUDA_ROOT}/lib64'.format(CUDA_ROOT=CUDA_ROOT))

    libraries.extend(['cuda', 'cudart'])

print 'include_dirs:', include_dirs
print 'library_dirs:', library_dirs
print 'libraries:', libraries

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
)

setup(
    name='nervana-aeon',
    version='0.1',
    packages=['aeon'],
    ext_modules=[module],
)
