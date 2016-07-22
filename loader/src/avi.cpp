#include "avi.hpp"

using namespace std;

namespace nervana
{

MjpegFileInputStream::MjpegFileInputStream() :
    m_is_valid(false),
    m_f()
{
}

MjpegFileInputStream::MjpegFileInputStream(const string& filename) :
    m_is_valid(false),
    m_f()
{
    open(filename);
}

bool MjpegFileInputStream::isOpened() const
{
    return m_f.is_open();
}

bool MjpegFileInputStream::open(const string& filename)
{
    close();

    m_f.open(filename, ios::in | ios::binary);

    m_is_valid = isOpened();

    return m_is_valid;
}

void MjpegFileInputStream::close()
{
    if(isOpened())
    {
        m_is_valid = false;

        m_f.close();
    }
}

MjpegInputStream& MjpegFileInputStream::read(char* buf, uint64_t count)
{
    if(isOpened())
    {
        m_f.read(buf, count);
        m_is_valid = m_f.good();
        if(!m_f) {
            m_f.clear();
        }
    }

    return *this;
}

MjpegInputStream& MjpegFileInputStream::seekg(uint64_t pos)
{
    m_f.seekg(pos, m_f.beg);
    m_is_valid = m_f.good();

    return *this;
}

uint64_t MjpegFileInputStream::tellg()
{
    return m_f.tellg();
}

MjpegFileInputStream::operator bool()
{
    return m_is_valid;
}

MjpegFileInputStream::~MjpegFileInputStream()
{
    close();
}


MjpegMemoryInputStream::MjpegMemoryInputStream(char* data, size_t size) :
    m_is_valid{true},
    m_wrapper{data,size},
    m_f{&m_wrapper}
{
}

MjpegMemoryInputStream::~MjpegMemoryInputStream()
{
}

MjpegInputStream& MjpegMemoryInputStream::read(char* buf, uint64_t count) {
    m_f.read(buf, count);
    m_is_valid = m_f.good();
    if(!m_f) {
        m_f.clear();
    }

    return *this;
}

MjpegInputStream& MjpegMemoryInputStream::seekg(uint64_t pos) {
    m_f.seekg(pos, m_f.beg);
    m_is_valid = m_f.good();

    return *this;
}

uint64_t MjpegMemoryInputStream::tellg() {
    return m_f.tellg();
}

bool MjpegMemoryInputStream::isOpened() const {
    return true;
}

bool MjpegMemoryInputStream::open(const std::string& filename) {
    return true;
}

void MjpegMemoryInputStream::close() {
}

MjpegMemoryInputStream::operator bool() {
    return m_is_valid;
}

}// namespace nervana

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::AviMainHeader& avih)
{
    is.read((char*)(&avih), sizeof(nervana::AviMainHeader));
    return is;
}

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::AviStreamHeader& strh)
{
    is.read((char*)(&strh), sizeof(nervana::AviStreamHeader));
    return is;
}

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::BitmapInfoHeader& bmph)
{
    is.read((char*)(&bmph), sizeof(nervana::BitmapInfoHeader));
    return is;
}

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::RiffList& riff_list)
{
    is.read((char*)(&riff_list), sizeof(riff_list));
    return is;
}

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::RiffChunk& riff_chunk)
{
    is.read((char*)(&riff_chunk), sizeof(riff_chunk));
    return is;
}

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::AviIndex& idx1)
{
    is.read((char*)(&idx1), sizeof(idx1));
    return is;
}
