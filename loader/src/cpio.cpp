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
    : _magic(070707)
    , _dev(0)
    , _ino(0)
    , _mode(0100644)
    , _uid(0)
    , _gid(0)
    , _nlink(0)
    , _rdev(0)
    , _namesize(0)
{
    memset((void*)_mtime, 0, 2 * sizeof(short));
    memset((void*)_filesize, 0, 2 * sizeof(short));
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
    read_single_value(ifs, &_magic);
    affirm(_magic == 070707, "CPIO header magic incorrect");
    read_single_value(ifs, &_dev);
    read_single_value(ifs, &_ino);
    read_single_value(ifs, &_mode);
    read_single_value(ifs, &_uid);
    read_single_value(ifs, &_gid);
    read_single_value(ifs, &_nlink);
    read_single_value(ifs, &_rdev);
    uint32_t mtime;
    read_single_value(ifs, &_mtime);
    loadDoubleShort(&mtime, _mtime);
    read_single_value(ifs, &_namesize);
    read_single_value(ifs, &_filesize);
    loadDoubleShort(fileSize, _filesize);
    // Skip over filename.
    ifs.seekg(_namesize, ifs.cur);
    readPadding(ifs, _namesize);
}

void cpio::record_header::write(ostream& ofs, uint32_t fileSize, const char* fileName)
{
    _namesize = strlen(fileName) + 1;
    write_single_value(ofs, &_magic);
    write_single_value(ofs, &_dev);
    write_single_value(ofs, &_ino);
    write_single_value(ofs, &_mode);
    write_single_value(ofs, &_uid);
    write_single_value(ofs, &_gid);
    write_single_value(ofs, &_nlink);
    write_single_value(ofs, &_rdev);
    time_t mtime;
    time(&mtime);
    saveDoubleShort(_mtime, mtime);
    write_single_value(ofs, &_mtime);
    write_single_value(ofs, &_namesize);
    saveDoubleShort(_filesize, fileSize);
    write_single_value(ofs, &_filesize);
    // Write filename.
    ofs.write((char*)fileName, _namesize);
    writePadding(ofs, _namesize);
}

cpio::header::header()
    : _formatVersion(FORMAT_VERSION)
    , _writerVersion(WRITER_VERSION)
    , _itemCount(0)
{
    memset(_dataType, 0, sizeof(_dataType));
    memset(_unused, 0, sizeof(_unused));
}

void cpio::header::read(istream& ifs)
{
    read_single_value(ifs, &_magic);
    if (strncmp(_magic, MAGIC_STRING, 4) != 0)
    {
        throw std::runtime_error("Unrecognized format\n");
    }
    read_single_value(ifs, &_formatVersion);
    read_single_value(ifs, &_writerVersion);
    read_single_value(ifs, &_dataType);
    read_single_value(ifs, &_itemCount);
    read_single_value(ifs, &_unused);
}

void cpio::header::write(ostream& ofs)
{
    ofs.write((char*)MAGIC_STRING, strlen(MAGIC_STRING));
    write_single_value(ofs, &_formatVersion);
    write_single_value(ofs, &_writerVersion);
    write_single_value(ofs, &_dataType);
    write_single_value(ofs, &_itemCount);
    write_single_value(ofs, &_unused);
}

cpio::trailer::trailer()
{
    memset(_unused, 0, sizeof(_unused));
}

void cpio::trailer::write(ostream& ofs)
{
    write_single_value(ofs, &_unused);
}

void cpio::trailer::read(istream& ifs)
{
    read_single_value(ifs, &_unused);
}

cpio::reader::reader()
{
}

cpio::reader::reader(istream* is)
{
    _is = is;
    readHeader();
}

void cpio::reader::readHeader()
{
    uint32_t fileSize;
    _recordHeader.read(*_is, &fileSize);
    if (fileSize != sizeof(_header))
    {
        stringstream ss;
        ss << "unexpected header size.  expected " << sizeof(_header);
        ss << " found " << fileSize;
        throw std::runtime_error(ss.str());
    }

    _header.read(*_is);
}

void cpio::reader::read(nervana::buffer_in& dest)
{
    uint32_t element_size;
    _recordHeader.read(*_is, &element_size);
    dest.read(*_is, element_size);
    readPadding(*_is, element_size);
}

void cpio::reader::read(vector<char>& dest)
{
    uint32_t element_size;
    _recordHeader.read(*_is, &element_size);
    dest.reserve(element_size);
    dest.resize(element_size);
    _is->read(dest.data(), dest.size());
    readPadding(*_is, element_size);
}

int cpio::reader::itemCount()
{
    return _header._itemCount;
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
    _ifs.open(fileName, istream::binary);
    if (_ifs)
    {
        _is = &_ifs;
        readHeader();
        rc = true;
    }

    return rc;
}

void cpio::file_reader::close()
{
    if (_ifs.is_open() == true)
    {
        _ifs.close();
    }
}

cpio::file_writer::~file_writer()
{
    close();
}

void cpio::file_writer::open(const std::string& fileName, const std::string& dataType)
{
    static_assert(sizeof(_header) == 64, "file header is not 64 bytes");
    _fileName = fileName;
    _tempName = fileName + ".tmp";
    _ofs.open(_tempName, ostream::binary);
    _recordHeader.write(_ofs, 64, "cpiohdr");
    _fileHeaderOffset = _ofs.tellp();
    memset(_header._dataType, ' ', sizeof(_header._dataType));
    memcpy(_header._dataType, dataType.c_str(), std::min(8, (int)dataType.length()));
    // This will be incomplete until the write on close()
    _header.write(_ofs);
}

void cpio::file_writer::close()
{
    if (_ofs.is_open() == true)
    {
        // Write the trailer.
        static_assert(sizeof(_trailer) == 16, "file trailer is not 16 bytes");
        _recordHeader.write(_ofs, 16, "cpiotlr");
        _trailer.write(_ofs);
        _recordHeader.write(_ofs, 0, CPIO_FOOTER);
        // Need to write back the max size values before cleaning up
        _ofs.seekp(_fileHeaderOffset, _ofs.beg);
        _header.write(_ofs);
        _ofs.close();
        int result = rename(_tempName.c_str(), _fileName.c_str());
        if (result != 0)
        {
            stringstream ss;
            ss << "Could not create " << _fileName;
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
    snprintf(fileName, sizeof(fileName), "rec_%07d.%02d", _header._itemCount, element_idx);
    _recordHeader.write(_ofs, elem_size, fileName);
    _ofs.write(elem, elem_size);
    writePadding(_ofs, elem_size);
}
