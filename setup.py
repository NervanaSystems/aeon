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
        'axon/src/cpio.cpp',
        'axon/src/buffer_in.cpp',
        'axon/src/buffer_out.cpp',
        'axon/src/pyBackendWrapper.cpp',
        'axon/src/pyLoader.cpp',
        'axon/src/api_pversion.cpp',
        'axon/src/etl_audio.cpp',
        'axon/src/noise_clips.cpp',
        'axon/src/specgram.cpp',
        'axon/src/log.cpp',
        'axon/src/codec.cpp',
        'axon/src/image.cpp',
        'axon/src/etl_pixel_mask.cpp',
        'axon/src/etl_image.cpp',
        'axon/src/etl_image_var.cpp',
        'axon/src/etl_bbox.cpp',
        'axon/src/etl_label_map.cpp',
        'axon/src/etl_char_map.cpp',
        'axon/src/etl_localization.cpp',
        'axon/src/etl_video.cpp',
        'axon/src/util.cpp',
        'axon/src/wav_data.cpp',
        'axon/src/manifest.cpp',
        'axon/src/provider_factory.cpp',
        'axon/src/batch_file_loader.cpp',
        'axon/src/batch_loader_cpio_cache.cpp',
        'axon/src/sequential_batch_iterator.cpp',
        'axon/src/shuffled_batch_iterator.cpp',
        'axon/src/minibatch_iterator.cpp',
        'axon/src/batch_loader.cpp',
        'axon/src/box.cpp',
        'axon/src/buffer_pool.cpp',
        'axon/src/buffer_pool_in.cpp',
        'axon/src/buffer_pool_out.cpp',
    ],
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
)

setup(
    name='axon',
    version='0.1',
    packages=['axon'],
    ext_modules=[module],
)
