#pragma once

#include <memory>
#include <inttypes.h>
#include <vector>
#include <cassert>

namespace nervana {
    class vector;
}

class nervana::vector
{
public:
    vector(size_t size);
    vector(vector& v,size_t offset, size_t length);

    static std::vector<nervana::vector> create(size_t count, size_t block_size);

    uint8_t& operator[](size_t index) {
        assert(index<length);
        return data.get()[offset+index];
    }

    const uint8_t& operator[](size_t index) const {
        return data.get()[offset+index];
    }

    size_t size() const { return length; }

private:
    std::shared_ptr<uint8_t> data;
    size_t                   offset;
    size_t                   length;
    size_t                   data_size;
};
