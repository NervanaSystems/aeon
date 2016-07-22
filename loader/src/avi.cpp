#include "avi.hpp"

using namespace std;

std::istream& operator >> (std::istream& is, nervana::AviMainHeader& avih)
{
    is.read((char*)(&avih), sizeof(nervana::AviMainHeader));
    return is;
}

std::istream& operator >> (std::istream& is, nervana::AviStreamHeader& strh)
{
    is.read((char*)(&strh), sizeof(nervana::AviStreamHeader));
    return is;
}

std::istream& operator >> (std::istream& is, nervana::BitmapInfoHeader& bmph)
{
    is.read((char*)(&bmph), sizeof(nervana::BitmapInfoHeader));
    return is;
}

std::istream& operator >> (std::istream& is, nervana::RiffList& riff_list)
{
    is.read((char*)(&riff_list), sizeof(riff_list));
    return is;
}

std::istream& operator >> (std::istream& is, nervana::RiffChunk& riff_chunk)
{
    is.read((char*)(&riff_chunk), sizeof(riff_chunk));
    return is;
}

std::istream& operator >> (std::istream& is, nervana::AviIndex& idx1)
{
    is.read((char*)(&idx1), sizeof(idx1));
    return is;
}

int nervana::CV_FOURCC(char c1, char c2, char c3, char c4)
{
    return CV_FOURCC_MACRO(c1, c2, c3, c4);
}

string nervana::fourccToString(uint32_t fourcc)
{
    stringstream ss;
    ss <<  (char)(fourcc & 255) << (char)((fourcc >> 8) & 255) << (char)((fourcc >> 16) & 255) << (char)((fourcc >> 24) & 255);
    return ss.str();
}
