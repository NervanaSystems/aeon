/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <stdexcept>
#include <iostream>

#include "bstream.hpp"

using namespace std;
using namespace nervana;

bstream_base::bstream_base() :
    endian{endian_t::LITTLE}
{
}

bstream_base::~bstream_base()
{
}

void bstream_base::set_endian(bstream_base::endian_t value)
{
    endian = value;
}

bstream_mem::bstream_mem(const char* _data, size_t size) :
    data{_data},
    data_size{size},
    offset{0}
{
}

bstream_mem::bstream_mem(const std::vector<uint8_t>& _data) :
    data{(const char*)_data.data()},
    data_size{_data.size()},
    offset{0}
{
}

bstream_mem::bstream_mem(const std::vector<char>& _data) :
    data{(const char*)_data.data()},
    data_size{_data.size()},
    offset{0}
{
}

bstream_mem::~bstream_mem()
{
}

uint8_t  bstream_mem::readU8()
{
    return get_next_byte();
}

uint16_t bstream_mem::readU16()
{
    uint16_t rc;
    if(endian == endian_t::BIG)
    {
        rc = (get_next_byte() << 8);
        rc |= get_next_byte();
    }
    else
    {
        rc = get_next_byte();
        rc |= (get_next_byte() << 8);
    }
    return rc;
}

uint32_t bstream_mem::readU32()
{
    uint32_t rc;
    if(endian == endian_t::BIG)
    {
        rc = (get_next_byte() << 24);
        rc = (get_next_byte() << 16);
        rc = (get_next_byte() <<  8);
        rc |= get_next_byte();
    }
    else
    {
        rc = get_next_byte();
        rc |= (get_next_byte() <<  8);
        rc |= (get_next_byte() << 16);
        rc |= (get_next_byte() << 24);
    }
    return rc;
}

uint64_t bstream_mem::readU64()
{
    uint64_t rc;
    if(endian == endian_t::BIG)
    {
        rc = ((uint64_t)get_next_byte() << 56);
        rc = ((uint64_t)get_next_byte() << 48);
        rc = ((uint64_t)get_next_byte() << 40);
        rc = ((uint64_t)get_next_byte() << 32);
        rc = ((uint64_t)get_next_byte() << 24);
        rc = ((uint64_t)get_next_byte() << 16);
        rc = ((uint64_t)get_next_byte() <<  8);
        rc |= (uint64_t)get_next_byte();
    }
    else
    {
        rc =   (uint64_t)get_next_byte();
        rc |= ((uint64_t)get_next_byte() <<  8);
        rc |= ((uint64_t)get_next_byte() << 16);
        rc |= ((uint64_t)get_next_byte() << 24);
        rc |= ((uint64_t)get_next_byte() << 32);
        rc |= ((uint64_t)get_next_byte() << 40);
        rc |= ((uint64_t)get_next_byte() << 48);
        rc |= ((uint64_t)get_next_byte() << 56);
    }
    return rc;
}

int8_t   bstream_mem::readS8()
{
    return get_next_byte();
}

int16_t  bstream_mem::readS16()
{
    return (int64_t)readU16();
}

int32_t  bstream_mem::readS32()
{
    return (int64_t)readU32();
}

int64_t  bstream_mem::readS64()
{
    return (int64_t)readU64();
}

float    bstream_mem::readF32()
{
    return (float)readU32();
}

double   bstream_mem::readF64()
{
    return (double)readU64();
}

void     bstream_mem::seek(size_t _offset)
{
    offset = _offset;
}

uint8_t* bstream_mem::read(uint8_t* target, size_t count, size_t channels)
{
    for(int i=0; i<count; i+=channels)
    {
        // channels are stored as RGB but opencv needs BGR
        for(int i=0; i<channels; i++)
        {
            target[channels-i-1] = readU8();
        }
        target += channels;
    }
    return target;
}

uint8_t bstream_mem::get_next_byte()
{
    if(offset >= data_size)
    {
        throw out_of_range("read past end of input buffer");
    }
    return data[offset++];
}
