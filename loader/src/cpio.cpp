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

#include "cpio.hpp"
#include "util.hpp"

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

void cpio::record_header::loadDoubleShort(uint32_t* dst, ushort src[2])
{
    *dst = ((uint32_t)src[0]) << 16 | (uint32_t)src[1];
}

void cpio::record_header::saveDoubleShort(ushort* dst, uint32_t src)
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
    loadDoubleShort(&mtime, m_mtime);
    read_single_value(ifs, &m_namesize);
    read_single_value(ifs, &m_filesize);
    loadDoubleShort(fileSize, m_filesize);
    // Skip over filename.
    ifs.seekg(m_namesize, ifs.cur);
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
    saveDoubleShort(m_mtime, mtime);
    write_single_value(ofs, &m_mtime);
    write_single_value(ofs, &m_namesize);
    saveDoubleShort(m_filesize, fileSize);
    write_single_value(ofs, &m_filesize);
    // Write filename.
    ofs.write((char*)fileName, m_namesize);
    writePadding(ofs, m_namesize);
}

cpio::header::header()
    : m_formatVersion(FORMAT_VERSION)
    , m_writerVersion(WRITER_VERSION)
    , m_itemCount(0)
{
    memset(m_dataType, 0, sizeof(m_dataType));
    memset(m_unused, 0, sizeof(m_unused));
}

void cpio::header::read(istream& ifs)
{
    read_single_value(ifs, &m_magic);
    if (strncmp(m_magic, MAGIC_STRING, 4) != 0)
    {
        throw std::runtime_error("Unrecognized format\n");
    }
    read_single_value(ifs, &m_formatVersion);
    read_single_value(ifs, &m_writerVersion);
    read_single_value(ifs, &m_dataType);
    read_single_value(ifs, &m_itemCount);
    read_single_value(ifs, &m_unused);
}

void cpio::header::write(ostream& ofs)
{
    ofs.write((char*)MAGIC_STRING, strlen(MAGIC_STRING));
    write_single_value(ofs, &m_formatVersion);
    write_single_value(ofs, &m_writerVersion);
    write_single_value(ofs, &m_dataType);
    write_single_value(ofs, &m_itemCount);
    write_single_value(ofs, &m_unused);
}

cpio::trailer::trailer()
{
    memset(m_unused, 0, sizeof(m_unused));
}

void cpio::trailer::write(ostream& ofs)
{
    write_single_value(ofs, &m_unused);
}

void cpio::trailer::read(istream& ifs)
{
    read_single_value(ifs, &m_unused);
}

cpio::reader::reader()
{
}

cpio::reader::reader(istream* is)
{
    m_is = is;
    readHeader();
}

void cpio::reader::readHeader()
{
    uint32_t fileSize;
    m_recordHeader.read(*m_is, &fileSize);
    if (fileSize != sizeof(m_header))
    {
        stringstream ss;
        ss << "unexpected header size.  expected " << sizeof(m_header);
        ss << " found " << fileSize;
        throw std::runtime_error(ss.str());
    }

    m_header.read(*m_is);
}

void cpio::reader::read(nervana::buffer_in& dest)
{
    uint32_t element_size;
    m_recordHeader.read(*m_is, &element_size);
    dest.read(*m_is, element_size);
    readPadding(*m_is, element_size);
}

void cpio::reader::read(vector<char>& dest)
{
    uint32_t element_size;
    m_recordHeader.read(*m_is, &element_size);
    dest.reserve(element_size);
    dest.resize(element_size);
    m_is->read(dest.data(), dest.size());
    readPadding(*m_is, element_size);
}

int cpio::reader::itemCount()
{
    return m_header.m_itemCount;
}

cpio::file_reader::file_reader()
{
}

cpio::file_reader::~file_reader()
{
    close();
}

bool cpio::file_reader::open(const string& fileName)
{
    // returns true if file was opened successfully.
    bool rc = false;
    m_ifs.open(fileName, istream::binary);
    if (m_ifs)
    {
        m_is = &m_ifs;
        readHeader();
        rc = true;
    }

    return rc;
}

void cpio::file_reader::close()
{
    if (m_ifs.is_open() == true)
    {
        m_ifs.close();
    }
}

cpio::file_writer::~file_writer()
{
    close();
}

void cpio::file_writer::open(const std::string& fileName, const std::string& dataType)
{
    static_assert(sizeof(m_header) == 64, "file header is not 64 bytes");
    m_fileName = fileName;
    m_tempName = fileName + ".tmp";
    m_ofs.open(m_tempName, ostream::binary);
    m_recordHeader.write(m_ofs, 64, "cpiohdr");
    m_fileHeaderOffset = m_ofs.tellp();
    memset(m_header.m_dataType, ' ', sizeof(m_header.m_dataType));
    memcpy(m_header.m_dataType, dataType.c_str(), std::min(8, (int)dataType.length()));
    // This will be incomplete until the write on close()
    m_header.write(m_ofs);
}

void cpio::file_writer::close()
{
    if (m_ofs.is_open() == true)
    {
        // Write the trailer.
        static_assert(sizeof(m_trailer) == 16, "file trailer is not 16 bytes");
        m_recordHeader.write(m_ofs, 16, "cpiotlr");
        m_trailer.write(m_ofs);
        m_recordHeader.write(m_ofs, 0, CPIO_FOOTER);
        // Need to write back the max size values before cleaning up
        m_ofs.seekp(m_fileHeaderOffset, m_ofs.beg);
        m_header.write(m_ofs);
        m_ofs.close();
        int result = rename(m_tempName.c_str(), m_fileName.c_str());
        if (result != 0)
        {
            stringstream ss;
            ss << "Could not create " << m_fileName;
            ss << ": " << strerror(result);
            throw std::runtime_error(ss.str());
        }
    }
}

void cpio::file_writer::write_all_records(nervana::buffer_in_array& buff)
{
    int num_records = buff[0]->get_item_count();
    for (int i = 0; i < num_records; ++i)
    {
        write_record(buff, i);
    }
}

void cpio::file_writer::write_record(nervana::buffer_in_array& buff, int record_idx)
{
    uint32_t element_idx = 0;
    for (auto b : buff)
    {
        const vector<char>& record_element = b->get_item(record_idx);
        write_record_element(record_element.data(), record_element.size(), element_idx++);
    }
    increment_record_count();
}

void cpio::file_writer::write_record_element(const char* elem, uint32_t elem_size, uint32_t element_idx)
{
    char fileName[16];
    snprintf(fileName, sizeof(fileName), "rec_%07d.%02d", m_header.m_itemCount, element_idx);
    m_recordHeader.write(m_ofs, elem_size, fileName);
    m_ofs.write(elem, elem_size);
    writePadding(m_ofs, elem_size);
}
