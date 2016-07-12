#pragma once

#include <iostream>
#include <cassert>

namespace nervana {

    // Unpack char array in little endian order
    template<typename T> T unpack_le(const char* data, int offset=0, int count=sizeof(T)) {
        T rc = 0;
        for(int i=0; i<count; i++) {
            rc += (unsigned char)data[offset+i] << (8*i);
        }
        return rc;
    }

    // Unpack char array in big endian order
    template<typename T> T unpack_be(const char* data, int offset=0, int count=sizeof(T)) {
        T rc = 0;
        for(int i=0; i<count; i++) {
            rc += (unsigned char)data[offset+i] << (8*(count-i-1));
        }
        return rc;
    }

    // Pack char array in little endian order
    template<typename T> void pack_le(char* data, T value, int offset=0, int count=sizeof(T)) {
        for(int i=0; i<count; i++) {
            data[offset+i] = (char)(value >> (8*i));
        }
    }

    // Pack char array in big endian order
    template<typename T> void pack_be(char* data, T value, int offset=0, int count=sizeof(T)) {
        for(int i=0; i<count; i++) {
            data[offset+i] = (char)(value >> (8*(count-i-1)));
        }
    }

    void dump( const void*, size_t );

    std::string tolower(const std::string& s);
}
