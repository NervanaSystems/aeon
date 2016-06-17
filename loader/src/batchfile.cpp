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

#include "batchfile.hpp"

using namespace std;

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

void RecordHeader::read(IfStream& ifs, uint* fileSize) {
    ifs.read(&_magic);
    assert(_magic == 070707);
    ifs.read(&_dev);
    ifs.read(&_ino);
    ifs.read(&_mode);
    ifs.read(&_uid);
    ifs.read(&_gid);
    ifs.read(&_nlink);
    ifs.read(&_rdev);
    uint mtime;
    ifs.read(&_mtime);
    loadDoubleShort(&mtime, _mtime);
    ifs.read(&_namesize);
    ifs.read(&_filesize);
    loadDoubleShort(fileSize, _filesize);
    // Skip over filename.
    ifs.seekg(_namesize, ifs.cur);
    ifs.readPadding(_namesize);
}

void RecordHeader::write(OfStream& ofs, uint fileSize, const char* fileName) {
    _namesize = strlen(fileName) + 1;
    ofs.write(&_magic);
    ofs.write(&_dev);
    ofs.write(&_ino);
    ofs.write(&_mode);
    ofs.write(&_uid);
    ofs.write(&_gid);
    ofs.write(&_nlink);
    ofs.write(&_rdev);
    time_t mtime;
    time(&mtime);
    saveDoubleShort(_mtime, mtime);
    ofs.write(&_mtime);
    ofs.write(&_namesize);
    saveDoubleShort(_filesize, fileSize);
    ofs.write(&_filesize);
    // Write filename.
    ofs.write((char*) fileName, _namesize);
    ofs.writePadding(_namesize);
}

BatchFileHeader::BatchFileHeader()
: _formatVersion(FORMAT_VERSION), _writerVersion(WRITER_VERSION),
  _itemCount(0), _maxDatumSize(0), _maxTargetSize(0),
  _totalDataSize(0), _totalTargetsSize(0) {
    memset(_dataType, 0, sizeof(_dataType));
    memset(_unused, 0, sizeof(_unused));
}

void BatchFileHeader::read(IfStream& ifs) {
    ifs.read(&_magic);
    if (strncmp(_magic, MAGIC_STRING, 4) != 0) {
        throw std::runtime_error("Unrecognized format\n");
    }
    ifs.read(&_formatVersion);
    ifs.read(&_writerVersion);
    ifs.read(&_dataType);
    ifs.read(&_itemCount);
    ifs.read(&_maxDatumSize);
    ifs.read(&_maxTargetSize);
    ifs.read(&_totalDataSize);
    ifs.read(&_totalTargetsSize);
    ifs.read(&_unused);
}

void BatchFileHeader::write(OfStream& ofs) {
    ofs.write((char*) MAGIC_STRING, strlen(MAGIC_STRING));
    ofs.write(&_formatVersion);
    ofs.write(&_writerVersion);
    ofs.write(&_dataType);
    ofs.write(&_itemCount);
    ofs.write(&_maxDatumSize);
    ofs.write(&_maxTargetSize);
    ofs.write(&_totalDataSize);
    ofs.write(&_totalTargetsSize);
    ofs.write(&_unused);
}

BatchFileTrailer::BatchFileTrailer() {
    memset(_unused, 0, sizeof(_unused));
}

void BatchFileTrailer::write(OfStream& ofs) {
    ofs.write(&_unused);
}

void BatchFileTrailer::read(IfStream& ifs) {
    ifs.read(&_unused);
}










BatchFileReader::BatchFileReader() {
}

BatchFileReader::BatchFileReader(const string& fileName) {
    if(!open(fileName)) {
        stringstream ss;
        ss << "couldn't open " << fileName;
        throw std::runtime_error(ss.str());
    }
}

BatchFileReader::~BatchFileReader() {
    close();
}

bool BatchFileReader::open(const string& fileName) {
    // returns true if file was opened successfully.
    assert(_ifs.is_open() == false);
    _ifs.open(fileName, IfStream::binary);
    if(!_ifs) {
        return false;
    }

    uint fileSize;
    _recordHeader.read(_ifs, &fileSize);
    if(fileSize != sizeof(_fileHeader)) {
        return false;
    }

    _fileHeader.read(_ifs);

    return true;
}

void BatchFileReader::close() {
    if (_ifs.is_open() == true) {
        _ifs.close();
    }
}

void BatchFileReader::read(Buffer& dest) {
    uint datumSize;
    _recordHeader.read(_ifs, &datumSize);
    dest.read(_ifs, datumSize);
    _ifs.readPadding(datumSize);
}

int BatchFileReader::itemCount() {
    return _fileHeader._itemCount;
}

int BatchFileReader::totalDataSize() {
    return _fileHeader._totalDataSize;
}

int BatchFileReader::totalTargetsSize() {
    return _fileHeader._totalTargetsSize;
}

int BatchFileReader::maxDatumSize() {
    return _fileHeader._maxDatumSize;
}

int BatchFileReader::maxTargetSize() {
    return _fileHeader._maxTargetSize;
}
















BatchFileWriter::~BatchFileWriter() {
    close();
}

void BatchFileWriter::open(const std::string& fileName, const std::string& dataType) {
    static_assert(sizeof(_fileHeader) == 64, "file header is not 64 bytes");
    _fileName = fileName;
    _tempName = fileName + ".tmp";
    assert(_ofs.is_open() == false);
    _ofs.open(_tempName, OfStream::binary);
    _recordHeader.write(_ofs, 64, "cpiohdr");
    _fileHeaderOffset = _ofs.tellp();
    memset(_fileHeader._dataType, ' ', sizeof(_fileHeader._dataType));
    memcpy(_fileHeader._dataType, dataType.c_str(),
           std::min(8, (int) dataType.length()));
    // This will be incomplete until the write on close()
    _fileHeader.write(_ofs);
}

void BatchFileWriter::close() {
    if (_ofs.is_open() == true) {
        // Write the trailer.
        static_assert(sizeof(_fileTrailer) == 16,
                      "file trailer is not 16 bytes");
        _recordHeader.write(_ofs, 16, "cpiotlr");
        _fileTrailer.write(_ofs);
        _recordHeader.write(_ofs, 0, CPIO_FOOTER);
        // Need to write back the max size values before cleaning up
        _ofs.seekp(_fileHeaderOffset, _ofs.beg);
        _fileHeader.write(_ofs);
        _ofs.close();
        int result = rename(_tempName.c_str(), _fileName.c_str());
        if (result != 0) {
            stringstream ss;
            ss << "Could not create " << _fileName;
            throw std::runtime_error(ss.str());
        }
    }
}

void BatchFileWriter::writeItem(char* datum, char* target,
               uint datumSize, uint targetSize) {
    char fileName[16];
    // Write the datum.
    sprintf(fileName, "cpiodtm%d",  _fileHeader._itemCount);
    _recordHeader.write(_ofs, datumSize, fileName);
    _ofs.write(datum, datumSize);
    _ofs.writePadding(datumSize);
    // Write the target.
    sprintf(fileName, "cpiotgt%d",  _fileHeader._itemCount);
    _recordHeader.write(_ofs, targetSize, fileName);
    _ofs.write(target, targetSize);
    _ofs.writePadding(targetSize);

    _fileHeader._maxDatumSize =
            std::max(datumSize, _fileHeader._maxDatumSize);
    _fileHeader._maxTargetSize =
            std::max(targetSize, _fileHeader._maxTargetSize);
    _fileHeader._totalDataSize += datumSize;
    _fileHeader._totalTargetsSize += targetSize;
    _fileHeader._itemCount++;
}

void BatchFileWriter::writeItem(ByteVect &datum, ByteVect &target) {
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
    std::ifstream ifs(filn, std::ifstream::binary);
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
