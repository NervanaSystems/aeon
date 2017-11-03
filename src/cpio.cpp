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

// https://people.freebsd.org/~kientzle/libarchive/man/cpio.5.txt

#include "cpio.hpp"
#include "util.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

namespace nervana
{
    template <typename T>
    void read_single_value(istream& ifs, T* data)
    {
        ifs.read(reinterpret_cast<char*>(data), sizeof(T));
    }

    void readPadding(istream& ifs, uint32_t length)
    {
        // Read a byte if length is odd.
        if (length % 2 != 0)
        {
            char byte = 0;
            read_single_value(ifs, &byte);
        }
    }

    template <typename T>
    void write_single_value(ostream& ofs, T* data)
    {
        ofs.write(reinterpret_cast<char*>(data), sizeof(T));
    }

    void writePadding(ostream& ofs, uint32_t length)
    {
        // Write a byte if length is odd.
        if (length % 2 != 0)
        {
            char byte = 0;
            write_single_value(ofs, &byte);
        }
    }
}

cpio::record_header::record_header()
    : m_magic(070707)
    , m_dev(0)
    , m_ino(0)
    , m_mode(0100644)
    , m_uid(0)
    , m_gid(0)
    , m_nlink(0)
    , m_rdev(0)
    , m_namesize(0)
{
    memset((void*)m_mtime, 0, 2 * sizeof(short));
    memset((void*)m_filesize, 0, 2 * sizeof(short));
}

void cpio::record_header::load_double_short(uint32_t* dst, ushort src[2])
{
    *dst = ((uint32_t)src[0]) << 16 | (uint32_t)src[1];
}

void cpio::record_header::save_double_short(ushort* dst, uint32_t src)
{
    dst[0] = (ushort)(src >> 16);
    dst[1] = (ushort)src;
}

void cpio::record_header::read(istream& ifs, uint32_t* fileSize)
{
    read_single_value(ifs, &m_magic);
    affirm(m_magic == 070707, "CPIO header magic incorrect");
    read_single_value(ifs, &m_dev);
    read_single_value(ifs, &m_ino);
    read_single_value(ifs, &m_mode);
    read_single_value(ifs, &m_uid);
    read_single_value(ifs, &m_gid);
    read_single_value(ifs, &m_nlink);
    read_single_value(ifs, &m_rdev);
    uint32_t mtime;
    read_single_value(ifs, &m_mtime);
    load_double_short(&mtime, m_mtime);
    read_single_value(ifs, &m_namesize);
    read_single_value(ifs, &m_filesize);
    load_double_short(fileSize, m_filesize);

    auto buffer = new char[m_namesize];
    ifs.read(buffer, m_namesize);
    m_filename = string(buffer, m_namesize - 1);
    delete[] buffer;

    readPadding(ifs, m_namesize);
}

void cpio::record_header::write(ostream& ofs, uint32_t fileSize, const char* fileName)
{
    m_namesize = strlen(fileName) + 1;
    write_single_value(ofs, &m_magic);
    write_single_value(ofs, &m_dev);
    write_single_value(ofs, &m_ino);
    write_single_value(ofs, &m_mode);
    write_single_value(ofs, &m_uid);
    write_single_value(ofs, &m_gid);
    write_single_value(ofs, &m_nlink);
    write_single_value(ofs, &m_rdev);
    time_t mtime;
    time(&mtime);
    save_double_short(m_mtime, mtime);
    write_single_value(ofs, &m_mtime);
    write_single_value(ofs, &m_namesize);
    save_double_short(m_filesize, fileSize);
    write_single_value(ofs, &m_filesize);
    // Write filename.
    ofs.write((char*)fileName, m_namesize);
    writePadding(ofs, m_namesize);
}

cpio::file_header::file_header()
    : m_format_version(FORMAT_VERSION)
    , m_writer_version(WRITER_VERSION)
    , m_record_count(0)
    , m_elements_per_record(0)
{
    memset(m_data_type, 0, sizeof(m_data_type));
    memset(m_unused, 0, sizeof(m_unused));
}

void cpio::file_header::read(istream& ifs)
{
    read_single_value(ifs, &m_magic);
    if (strncmp(m_magic, MAGIC_STRING, 4) != 0)
    {
        throw std::runtime_error("Unrecognized format\n");
    }
    read_single_value(ifs, &m_format_version);
    read_single_value(ifs, &m_writer_version);
    read_single_value(ifs, &m_data_type);
    read_single_value(ifs, &m_record_count);
    read_single_value(ifs, &m_elements_per_record);
    read_single_value(ifs, &m_unused);
}

void cpio::file_header::write(ostream& ofs)
{
    ofs.write(MAGIC_STRING, strlen(MAGIC_STRING));
    write_single_value(ofs, &m_format_version);
    write_single_value(ofs, &m_writer_version);
    write_single_value(ofs, &m_data_type);
    write_single_value(ofs, &m_record_count);
    write_single_value(ofs, &m_elements_per_record);
    write_single_value(ofs, &m_unused);
}

cpio::file_trailer::file_trailer()
{
    memset(m_unused, 0, sizeof(m_unused));
}

void cpio::file_trailer::write(ostream& ofs)
{
    write_single_value(ofs, &m_unused);
}

void cpio::file_trailer::read(istream& ifs)
{
    read_single_value(ifs, &m_unused);
}

cpio::reader::reader(istream& is)
    : m_is{is}
{
    read_header();
}

cpio::reader::~reader()
{
    close();
}

void cpio::reader::read_header()
{
    uint32_t fileSize;
    m_record_header.read(m_is, &fileSize);
    if (fileSize != sizeof(m_header))
    {
        stringstream ss;
        ss << "unexpected header size.  expected " << sizeof(m_header);
        ss << " found " << fileSize;
        throw std::runtime_error(ss.str());
    }

    m_header.read(m_is);
}

void cpio::reader::read(nervana::encoded_record_list& dest, size_t element_count)
{
    encoded_record record;
    for (size_t i = 0; i < element_count; i++)
    {
        vector<char> buffer;
        read(buffer);
        record.add_element(buffer);
    }
    dest.add_record(record);
}

string cpio::reader::read(vector<char>& dest)
{
    uint32_t element_size;
    m_record_header.read(m_is, &element_size);
    dest.reserve(element_size);
    dest.resize(element_size);
    m_is.read(dest.data(), dest.size());
    readPadding(m_is, element_size);
    return m_record_header.m_filename;
}

int cpio::reader::record_count()
{
    return m_header.m_record_count;
}

void cpio::reader::close()
{
}

cpio::writer::writer(ostream& stream)
    : m_ofs{stream}
{
    string dataType = "";
    static_assert(sizeof(m_header) == 64, "file header is not 64 bytes");
    m_recordHeader.write(m_ofs, 64, AEON_HEADER);
    m_fileHeaderOffset = m_ofs.tellp();
    memset(m_header.m_data_type, ' ', sizeof(m_header.m_data_type));
    memcpy(m_header.m_data_type, dataType.c_str(), std::min(8, (int)dataType.length()));
    // This will be incomplete until the write on close()
    m_header.write(m_ofs);
}

cpio::writer::~writer()
{
    if (m_ofs)
    {
        // Write the trailer.
        static_assert(sizeof(m_trailer) == 16, "file trailer is not 16 bytes");
        m_recordHeader.write(m_ofs, 16, AEON_TRAILER);
        m_trailer.write(m_ofs);
        m_recordHeader.write(m_ofs, 0, CPIO_TRAILER);
        // Need to write back the max size values before cleaning up
        m_ofs.seekp(m_fileHeaderOffset, m_ofs.beg);
        m_header.write(m_ofs);
    }
}

void cpio::writer::write_all_records(const nervana::encoded_record_list& buff)
{
    size_t record_count  = buff.size();
    size_t element_index = 0;
    if (m_header.m_elements_per_record == 0)
    {
        m_header.m_elements_per_record = record_count;
    }
    for (auto b : buff)
    {
        for (auto element : b)
        {
            write_record_element(element.data(), element.size(), element_index++);
        }
        increment_record_count();
    }
}

void cpio::writer::write_record_element(const char* element,
                                        uint32_t    element_size,
                                        uint32_t    element_index)
{
    char file_name[16];
    snprintf(file_name, sizeof(file_name), "rec_%07d.%02d", m_header.m_record_count, element_index);
    m_recordHeader.write(m_ofs, element_size, file_name);
    m_ofs.write(element, element_size);
    writePadding(m_ofs, element_size);
}
