#!/bin/bash

export CFLAGS="-Wno-deprecated-declarations -std=c++11"
export CC="clang++"

pkg-config --exists opencv
if [[ $? == 0 ]]; then
    export IMGFLAG="-DHAS_IMGLIB"
	export INC="$(pkg-config --cflags opencv)"
	export IMGLDIR="$(pkg-config --libs-only-L opencv)"
	export IMGLIBS="$(pkg-config --libs-only-l opencv)"
fi

pkg-config --exists libavutil libavformat libavcodec libswscale
if [[ $? == 0 ]]; then
	export VIDFLAG="-DHAS_VIDLIB"
	export AUDFLAG="-DHAS_AUDLIB"
    export VIDLDIR="$(pkg-config --libs-only-L libavutil libavformat libavcodec libswscale)"
    export VIDLIBS="-lavutil -lavformat -lavcodec -lswscale"
fi

export MEDIAFLAGS="${IMGFLAG} ${VIDFLAG} ${AUDFLAG}"
export LDIR="${IMGLDIR} ${VIDLDIR}"
export LIBS="${IMGLIBS} ${VIDLIBS}"

export INC="-I$(python -c 'from distutils.sysconfig import get_python_inc; print get_python_inc()') ${INC}"
export INC="-I$(python -c 'import numpy; print numpy.get_include()') ${INC}"
export LIBS="-lpython2.7 ${LIBS}"

if [ "${HAS_GPU}" = true ] ; then
    if [ -z "${CUDA_ROOT}" ] ; then
        export CUDA_ROOT="${which nvcc | sed 's|/bin/nvcc||g'}"
    fi

	export GPUFLAG="-DHAS_GPU"
	export INC="-I${CUDA_ROOT}/include ${INC}"
	if [ "$(uname -s)" = "Darwin" ] ; then
		export LDIR="-L${CUDA_ROOT}/lib ${LDIR}"
	else
		export LDIR="-L${CUDA_ROOT}/lib64 ${LDIR}"
    fi
	export LIBS="-lcuda -lcudart ${LIBS}"
fi

