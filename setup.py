import os
import sys
from distutils.core import setup, Extension
import subprocess

for line in subprocess.check_output('bash -c "source env.sh; env"',
        shell=True).strip().split(b'\n'):
    key, _, value = line.partition(b"=")
    if sys.version_info.major == 2:
        os.environ[key] = value
    else:
        os.environ[key.decode('utf-8')] = value.decode('utf-8')

include_dirs = os.environ['INC'].replace('-I', '').split()
library_dirs = os.environ['LDIR'].replace('-L', '').split()
libraries = os.environ['LIBS'].replace('-l', '').split()
extra_compile_args = os.environ['CFLAGS'].split()
sources = [
    'loader/src/'+src for src in os.environ['SRCS'].split()
]

# Parallel build from http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool().imap(_single_compile,objects))
    return objects
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile

module = Extension(
    'aeon',
    sources=sources,
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
)

setup(
    name='nervana-aeon',
    version='0.2.6',
    ext_modules=[module],
)
