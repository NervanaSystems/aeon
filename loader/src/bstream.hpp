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

#pragma once

#include <vector>
#include <cstddef>
#include <cinttypes>

namespace nervana
{
    class bstream_base;
    class bstream_mem;
}

class nervana::bstream_base
{
public:
    enum class endian_t
    {
        BIG,
        LITTLE
    };
    bstream_base();
    virtual ~bstream_base();
    void set_endian(endian_t);

    virtual uint8_t  readU8()  = 0;
    virtual uint16_t readU16() = 0;
    virtual uint32_t readU32() = 0;
    virtual uint64_t readU64() = 0;
    virtual int8_t   readS8()  = 0;
    virtual int16_t  readS16() = 0;
    virtual int32_t  readS32() = 0;
    virtual int64_t  readS64() = 0;
    virtual float    readF32() = 0;
    virtual double   readF64() = 0;

    virtual void     seek(size_t offset) = 0;
    virtual uint8_t* read(uint8_t* target, size_t count, size_t channels) = 0;
protected:
    endian_t endian = endian_t::LITTLE;
};

class nervana::bstream_mem : public bstream_base
{
public:
    bstream_mem(const char* data, size_t size);
    bstream_mem(const std::vector<uint8_t>& data);
    bstream_mem(const std::vector<char>& data);
    ~bstream_mem();

    uint8_t  readU8()  override;
    uint16_t readU16() override;
    uint32_t readU32() override;
    uint64_t readU64() override;
    int8_t   readS8()  override;
    int16_t  readS16() override;
    int32_t  readS32() override;
    int64_t  readS64() override;
    float    readF32() override;
    double   readF64() override;

    void     seek(size_t offset) override;
    uint8_t* read(uint8_t* target, size_t count, size_t channels) override;

private:
    bstream_mem() = delete;

    uint8_t get_next_byte();

    const char* data;
    size_t      data_size;
    size_t      offset;
};
