#include "vector.hpp"

using namespace std;

nervana::vector::vector(size_t size) :
    offset{0},
    length{size},
    data_size{size}
{
    data = shared_ptr<uint8_t>(new uint8_t[size],std::default_delete<uint8_t[]>());
}

nervana::vector::vector(vector& v,size_t _offset, size_t _length) :
    data{v.data},
    offset{_offset},
    length{_length},
    data_size{v.data_size}
{

}

std::vector<nervana::vector> nervana::vector::create(size_t count, size_t block_size) {
    std::vector<vector> rc;
    vector mother(count * block_size);
    for (int i=0; i<count; i++) {
        rc.emplace_back(mother,block_size*i,block_size);
    }
    return rc;
}
