#pragma once

#include <iostream>
#include <cassert>

namespace nervana {
    // Unpack char array in little endian order
    template<typename T> T unpack_le(const char* data, int offset=0, int count=sizeof(T)) {
    //    std::assert(count<=(int)sizeof(T));
        T rc = 0;
        for(int i=0; i<count; i++) {
            rc += data[offset+i] << (8*i);
        }
        return rc;
    }

    // Unpack char array in big endian order
    template<typename T> T unpack_be(const char* data, int offset=0, int count=sizeof(T)) {
    //    std::assert(count<=(int)sizeof(T));
        T rc = 0;
        for(int i=0; i<count; i++) {
            rc += data[offset+i] << (8*(count-i-1));
        }
        return rc;
    }
}
