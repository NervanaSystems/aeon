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

using namespace std;

template <typename T>
void read_single_value(istream& ifs, T* data) {
    ifs.read(reinterpret_cast<char*>(data), sizeof(T));
}

void readPadding(istream& ifs, uint length) {
    // Read a byte if length is odd.
    if (length % 2 == 0) {
        return;
    }
    char byte = 0;
    read_single_value(ifs, &byte);
}

template <typename T>
void write_single_value(ostream& ofs, T* data) {
    ofs.write(reinterpret_cast<char*>(data), sizeof(T));
}

void writePadding(ostream& ofs, uint length) {
    // Write a byte if length is odd.
    if (length % 2 == 0) {
        return;
    }
    char byte = 0;
    write_single_value(ofs, &byte);
}

RecordHeader::RecordHeader()
: _magic(070707), _dev(0), _ino(0), _mode(0100644), _uid(0), _gid(0),
  _nlink(0), _rdev(0), _namesize(0) {
    memset((void*) _mtime, 0, 2 * sizeof(short));
    memset((void*) _filesize, 0, 2 * sizeof(short));
}

void RecordHeader::loadDoubleShort(uint* dst, ushort src[2]) {
    *dst =  ((uint) src[0]) << 16 | (uint) src[1];
}

void RecordHeader::saveDoubleShort(ushort* dst, uint src) {
    dst[0] = (ushort) (src >> 16);
    dst[1] = (ushort) src;
}

void RecordHeader::read(istream& ifs, uint* fileSize) {
    read_single_value(ifs, &_magic);
    assert(_magic == 070707);
    read_single_value(ifs, &_dev);
    read_single_value(ifs, &_ino);
    read_single_value(ifs, &_mode);
    read_single_value(ifs, &_uid);
    read_single_value(ifs, &_gid);
    read_single_value(ifs, &_nlink);
    read_single_value(ifs, &_rdev);
    uint mtime;
    read_single_value(ifs, &_mtime);
    loadDoubleShort(&mtime, _mtime);
    read_single_value(ifs, &_namesize);
    read_single_value(ifs, &_filesize);
    loadDoubleShort(fileSize, _filesize);
    // Skip over filename.
    ifs.seekg(_namesize, ifs.cur);
    readPadding(ifs, _namesize);
}

void RecordHeader::write(ostream& ofs, uint fileSize, const char* fileName) {
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
    ofs.write((char*) fileName, _namesize);
    writePadding(ofs, _namesize);
}

CPIOHeader::CPIOHeader()
: _formatVersion(FORMAT_VERSION), _writerVersion(WRITER_VERSION),
  _itemCount(0), _maxDatumSize(0), _maxTargetSize(0),
  _totalDataSize(0), _totalTargetsSize(0) {
    memset(_dataType, 0, sizeof(_dataType));
    memset(_unused, 0, sizeof(_unused));
}

void CPIOHeader::read(istream& ifs) {
    read_single_value(ifs, &_magic);
    if (strncmp(_magic, MAGIC_STRING, 4) != 0) {
        throw std::runtime_error("Unrecognized format\n");
    }
    read_single_value(ifs, &_formatVersion);
    read_single_value(ifs, &_writerVersion);
    read_single_value(ifs, &_dataType);
    read_single_value(ifs, &_itemCount);
    read_single_value(ifs, &_maxDatumSize);
    read_single_value(ifs, &_maxTargetSize);
    read_single_value(ifs, &_totalDataSize);
    read_single_value(ifs, &_totalTargetsSize);
    read_single_value(ifs, &_unused);
}

void CPIOHeader::write(ostream& ofs) {
    ofs.write((char*) MAGIC_STRING, strlen(MAGIC_STRING));
    write_single_value(ofs, &_formatVersion);
    write_single_value(ofs, &_writerVersion);
    write_single_value(ofs, &_dataType);
    write_single_value(ofs, &_itemCount);
    write_single_value(ofs, &_maxDatumSize);
    write_single_value(ofs, &_maxTargetSize);
    write_single_value(ofs, &_totalDataSize);
    write_single_value(ofs, &_totalTargetsSize);
    write_single_value(ofs, &_unused);
}

CPIOTrailer::CPIOTrailer() {
    memset(_unused, 0, sizeof(_unused));
}

void CPIOTrailer::write(ostream& ofs) {
    write_single_value(ofs, &_unused);
}

void CPIOTrailer::read(istream& ifs) {
    read_single_value(ifs, &_unused);
}








CPIOReader::CPIOReader() {
}

CPIOReader::CPIOReader(istream* is) {
    _is = is;
    readHeader();
}

void CPIOReader::readHeader() {
    uint fileSize;
    _recordHeader.read(*_is, &fileSize);
    if(fileSize != sizeof(_header)) {
        stringstream ss;
        ss << "unexpected header size.  expected " << sizeof(_header);
        ss << " found " << fileSize;
        throw std::runtime_error(ss.str());
    }

    _header.read(*_is);
}

void CPIOReader::read(Buffer& dest) {
    uint datumSize;
    _recordHeader.read(*_is, &datumSize);
    dest.read(*_is, datumSize);
    readPadding(*_is, datumSize);
}

int CPIOReader::itemCount() {
    return _header._itemCount;
}

int CPIOReader::totalDataSize() {
    return _header._totalDataSize;
}

int CPIOReader::totalTargetsSize() {
    return _header._totalTargetsSize;
}

int CPIOReader::maxDatumSize() {
    return _header._maxDatumSize;
}

int CPIOReader::maxTargetSize() {
    return _header._maxTargetSize;
}

CPIOFileReader::CPIOFileReader() {
}

CPIOFileReader::~CPIOFileReader() {
    close();
}

bool CPIOFileReader::open(const string& fileName) {
    // returns true if file was opened successfully.
    assert(_ifs.is_open() == false);

    _ifs.open(fileName, istream::binary);
    if(!_ifs) {
        return false;
    }

    _is = &_ifs;

    readHeader();

    return true;
}

void CPIOFileReader::close() {
    if (_ifs.is_open() == true) {
        _ifs.close();
    }
}
















CPIOFileWriter::~CPIOFileWriter() {
    close();
}

void CPIOFileWriter::open(const std::string& fileName, const std::string& dataType) {
    static_assert(sizeof(_header) == 64, "file header is not 64 bytes");
    _fileName = fileName;
    _tempName = fileName + ".tmp";
    assert(_ofs.is_open() == false);
    _ofs.open(_tempName, ostream::binary);
    _recordHeader.write(_ofs, 64, "cpiohdr");
    _fileHeaderOffset = _ofs.tellp();
    memset(_header._dataType, ' ', sizeof(_header._dataType));
    memcpy(_header._dataType, dataType.c_str(),
           std::min(8, (int) dataType.length()));
    // This will be incomplete until the write on close()
    _header.write(_ofs);
}

void CPIOFileWriter::close() {
    if (_ofs.is_open() == true) {
        // Write the trailer.
        static_assert(sizeof(_trailer) == 16,
                      "file trailer is not 16 bytes");
        _recordHeader.write(_ofs, 16, "cpiotlr");
        _trailer.write(_ofs);
        _recordHeader.write(_ofs, 0, CPIO_FOOTER);
        // Need to write back the max size values before cleaning up
        _ofs.seekp(_fileHeaderOffset, _ofs.beg);
        _header.write(_ofs);
        _ofs.close();
        int result = rename(_tempName.c_str(), _fileName.c_str());
        if (result != 0) {
            stringstream ss;
            ss << "Could not create " << _fileName;
            throw std::runtime_error(ss.str());
        }
    }
}

void CPIOFileWriter::writeItem(char* datum, char* target,
               uint datumSize, uint targetSize) {
    char fileName[16];
    // Write the datum.
    sprintf(fileName, "cpiodtm%d",  _header._itemCount);
    _recordHeader.write(_ofs, datumSize, fileName);
    _ofs.write(datum, datumSize);
    writePadding(_ofs, datumSize);
    // Write the target.
    sprintf(fileName, "cpiotgt%d",  _header._itemCount);
    _recordHeader.write(_ofs, targetSize, fileName);
    _ofs.write(target, targetSize);
    writePadding(_ofs, targetSize);

    _header._maxDatumSize =
            std::max(datumSize, _header._maxDatumSize);
    _header._maxTargetSize =
            std::max(targetSize, _header._maxTargetSize);
    _header._totalDataSize += datumSize;
    _header._totalTargetsSize += targetSize;
    _header._itemCount++;
}

void CPIOFileWriter::writeItem(ByteVect &datum, ByteVect &target) {
    uint    datumSize = datum.size();
    uint    targetSize = target.size();
    writeItem(&datum[0], &target[0], datumSize, targetSize);
}









// Some utilities that would be used by batch writers
int readFileLines(const string &filn, LineList &ll) {
    std::ifstream ifs(filn);
    if (ifs) {
        for (string line; std::getline( ifs, line ); /**/ )
           ll.push_back( line );
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}

int readFileBytes(const string &filn, ByteVect &b) {
/* Reads in the binary file as a sequence of bytes, resizing
 * the provided byte vector to fit
*/
    std::ifstream ifs(filn, std::istream::binary);
    if (ifs) {
        ifs.seekg (0, ifs.end);
        int length = ifs.tellg();
        ifs.seekg (0, ifs.beg);

        b.resize(length);
        ifs.read(&b[0], length);
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}
