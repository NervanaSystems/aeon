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

#include <time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>

#include "buffer_in.hpp"

#define FORMAT_VERSION 1
#define WRITER_VERSION 1
#define MAGIC_STRING "MACR"
#define CPIO_FOOTER "TRAILER!!!"

namespace nervana
{
    namespace cpio
    {
        class record_header;
        class header;
        class trailer;
        class reader;
        class file_reader;
        class file_writer;
    }
}

/*

The data is stored as a cpio archive and may be unpacked using the
GNU cpio utility.

    https://www.gnu.org/software/cpio/

Individual items are packed into a macrobatch file as follows:
    - header
    - datum 1
    - target 1
    - datum 2
    - target 2
      ...
    - trailer

Each of these items comprises of a cpio header record followed by data.

*/

class nervana::cpio::record_header
{
public:
    record_header();
    void loadDoubleShort(uint32_t* dst, uint16_t src[2]);

    void saveDoubleShort(uint16_t* dst, uint32_t src);

    void read(std::istream& ifs, uint32_t* fileSize);

    void write(std::ostream& ofs, uint32_t fileSize, const char* fileName);

public:
    uint16_t _magic;
    uint16_t _dev;
    uint16_t _ino;
    uint16_t _mode;
    uint16_t _uid;
    uint16_t _gid;
    uint16_t _nlink;
    uint16_t _rdev;
    uint16_t _mtime[2];
    uint16_t _namesize;
    uint16_t _filesize[2];
};

class nervana::cpio::header
{
    friend class reader;
    friend class file_writer;

public:
    header();
    void read(std::istream& ifs);
    void write(std::ostream& ofs);

private:
#pragma pack(1)
    char     _magic[4];
    uint32_t _formatVersion;
    uint32_t _writerVersion;
    char     _dataType[8];
    uint32_t _itemCount;
    uint8_t  _unused[40];
#pragma pack()
};

class nervana::cpio::trailer
{
public:
    trailer();
    void write(std::ostream& ofs);
    void read(std::istream& ifs);

private:
    uint32_t _unused[4];
};

class nervana::cpio::reader
{
public:
    reader();
    reader(std::istream* is);

    void read(nervana::buffer_in& dest);
    void read(std::vector<char>& dest);

    int itemCount();

protected:
    void readHeader();

    std::istream* _is;

    header        _header;
    trailer       _trailer;
    record_header _recordHeader;
};

/*
 * CPIOFileReader wraps file opening around the more generic CPIOReader
 * which only deals in istreams
 */

class nervana::cpio::file_reader : public reader
{
public:
    file_reader();
    ~file_reader();

    bool open(const std::string& fileName);
    void close();

private:
    std::ifstream _ifs;
};

class nervana::cpio::file_writer
{
public:
    ~file_writer();

    void open(const std::string& fileName, const std::string& dataType = "");
    void close();

    void write_all_records(nervana::buffer_in_array& buff);
    void write_record(nervana::buffer_in_array& buff, int record_idx);
    void write_record_element(const char* elem, uint32_t elem_size, uint32_t element_idx);
    void increment_record_count() { _header._itemCount++; }
private:
    std::ofstream _ofs;
    header        _header;
    trailer       _trailer;
    record_header _recordHeader;
    int           _fileHeaderOffset;
    std::string   _fileName;
    std::string   _tempName;
};
