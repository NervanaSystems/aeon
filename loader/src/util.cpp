#include <algorithm>
#include <string>

#include "util.hpp"

void nervana::dump( const void* _data, size_t _size ) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    int len = _size;
    assert(len % 16 == 0);
    int index = 0;
    while (index < len) {
        printf("%08x", index);
        for (int i = 0; i < 8; i++) {
            printf(" %02x", data[i]);
        }
        printf("  ");
        for (int i = 8; i < 16; i++) {
            printf(" %02x", data[i]);
        }
        printf(" ");
        for (int i = 0; i < 16; i++) {
            printf("%c", (data[i] < 32)? '.' : data[i]);
        }
        printf("\n");
        data += 16;
        index += 16;
    }
}

std::string nervana::tolower(const std::string& s) {
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}
