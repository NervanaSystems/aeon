from cffi import FFI

ffi = FFI()

lib = ffi.dlopen('./libsomelib.so')
print('Loaded lib {0}'.format(lib))

# Describe the data type and function prototype to cffi.
with open('./im_struct.h') as f:
    ffi.cdef(f.read())

print('Calling image param loading')
dout = lib.default_image(230, 224)

print('dout = {0}, {1}, {2}'.format(dout._height, dout._width, dout._scaleMin))
