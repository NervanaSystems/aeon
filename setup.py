import os
from distutils.core import setup, Extension
import subprocess


def shell_stdout(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].strip()

for line in shell_stdout('bash -c "source env.sh; env"').split('\n'):
    key, _, value = line.partition("=")
    os.environ[key] = value

include_dirs = os.environ['INC'].replace('-I', '').split()
library_dirs = os.environ['LDIR'].replace('-L', '').split()
libraries = os.environ['LIBS'].replace('-l', '').split()
extra_compile_args = os.environ['CFLAGS'].split()
sources = [
    'loader/src/'+src for src in os.environ['SRCS'].split()
]

module = Extension(
    'aeon_lib',
    sources=sources,
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
