/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include "buffer_batch.hpp"

namespace nervana
{
    namespace cpio
    {
        static const uint32_t FORMAT_VERSION = 1;
        static const uint32_t WRITER_VERSION = 1;
        static const char*    MAGIC_STRING   = "MACR";
        static const char*    CPIO_TRAILER   = "TRAILER!!!";
        static const char*    AEON_HEADER    = "cpiohdr";
        static const char*    AEON_TRAILER   = "cpiotlr";

        class record_header;
        class file_header;
        class file_trailer;
        class reader;
        class writer;
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
    void load_double_short(uint32_t* dst, uint16_t src[2]);

    void save_double_short(uint16_t* dst, uint32_t src);

    void read(std::istream& ifs, uint32_t* fileSize);

    void write(std::ostream& ofs, uint32_t fileSize, const char* fileName);

public:
    uint16_t    m_magic;
    uint16_t    m_dev;
    uint16_t    m_ino;
    uint16_t    m_mode;
    uint16_t    m_uid;
    uint16_t    m_gid;
    uint16_t    m_nlink;
    uint16_t    m_rdev;
    uint16_t    m_mtime[2];
    uint16_t    m_namesize;
    uint16_t    m_filesize[2];
    std::string m_filename;
};

class nervana::cpio::file_header
{
    friend class reader;
    friend class writer;

public:
    file_header();
    void read(std::istream& ifs);
    void write(std::ostream& ofs);

private:
#pragma pack(1)
    char     m_magic[4];
    uint32_t m_format_version;
    uint32_t m_writer_version;
    char     m_data_type[8];
    uint32_t m_record_count;
    uint32_t m_elements_per_record;
    uint8_t  m_unused[36];
#pragma pack()
};

class nervana::cpio::file_trailer
{
public:
    file_trailer();
    void write(std::ostream& ofs);
    void read(std::istream& ifs);

private:
    uint32_t m_unused[4];
};

class nervana::cpio::reader
{
public:
    reader(std::istream& is);
    virtual ~reader();

    void close();
    void read(nervana::encoded_record_list& dest, size_t element_count);
    std::string read(std::vector<char>& dest);

    int record_count();

protected:
    void read_header();

    std::istream& m_is;

    file_header   m_header;
    file_trailer  m_trailer;
    record_header m_record_header;
};

class nervana::cpio::writer
{
public:
    writer(std::ostream& stream);
    virtual ~writer();

    void write_all_records(const nervana::encoded_record_list& buff);
    void write_record_element(const char* elem, uint32_t elem_size, uint32_t element_idx);
    void increment_record_count() { m_header.m_record_count++; }
private:
    std::ostream& m_ofs;

    file_header   m_header;
    file_trailer  m_trailer;
    record_header m_recordHeader;
    int           m_fileHeaderOffset;
    std::string   m_fileName;
    std::string   m_tempName;
};
