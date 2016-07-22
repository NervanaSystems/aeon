#pragma once

#include <deque>
#include <string>
#include <fstream>
#include <sstream>
#include <cinttypes>

#include "util.hpp"

namespace nervana {
    class MjpegInputStream;
    class MjpegFileInputStream;
    class MjpegMemoryInputStream;

#pragma pack(push, 1)
struct AviMainHeader
{
    uint32_t dwMicroSecPerFrame;    //  The period between video frames
    uint32_t dwMaxBytesPerSec;      //  Maximum data rate of the file
    uint32_t dwReserved1;           // 0
    uint32_t dwFlags;               //  0x10 AVIF_HASINDEX: The AVI file has an idx1 chunk containing an index at the end of the file.
    uint32_t dwTotalFrames;         // Field of the main header specifies the total number of frames of data in file.
    uint32_t dwInitialFrames;       // Is used for interleaved files
    uint32_t dwStreams;             // Specifies the number of streams in the file.
    uint32_t dwSuggestedBufferSize; // Field specifies the suggested buffer size forreading the file
    uint32_t dwWidth;               // Fields specify the width of the AVIfile in pixels.
    uint32_t dwHeight;              // Fields specify the height of the AVIfile in pixels.
    uint32_t dwReserved[4];         // 0, 0, 0, 0
};

struct AviStreamHeader
{
    uint32_t fccType;              // 'vids', 'auds', 'txts'...
    uint32_t fccHandler;           // "cvid", "DIB "
    uint32_t dwFlags;               // 0
    uint32_t dwPriority;            // 0
    uint32_t dwInitialFrames;       // 0
    uint32_t dwScale;               // 1
    uint32_t dwRate;                // Fps (dwRate - frame rate for video streams)
    uint32_t dwStart;               // 0
    uint32_t dwLength;              // Frames number (playing time of AVI file as defined by scale and rate)
    uint32_t dwSuggestedBufferSize; // For reading the stream
    uint32_t dwQuality;             // -1 (encoding quality. If set to -1, drivers use the default quality value)
    uint32_t dwSampleSize;          // 0 means that each frame is in its own chunk
    struct {
        int16_t left;
        int16_t top;
        int16_t right;
        int16_t bottom;
    } rcFrame;                // If stream has a different size than dwWidth*dwHeight(unused)
};

struct AviIndex
{
    uint32_t ckid;
    uint32_t dwFlags;
    uint32_t dwChunkOffset;
    uint32_t dwChunkLength;
};

struct BitmapInfoHeader
{
    uint32_t biSize;                // Write header size of BITMAPINFO header structure
    int32_t  biWidth;               // width in pixels
    int32_t  biHeight;              // heigth in pixels
    uint16_t biPlanes;              // Number of color planes in which the data is stored
    uint16_t biBitCount;            // Number of bits per pixel
    uint32_t biCompression;         // Type of compression used (uncompressed: NO_COMPRESSION=0)
    uint32_t biSizeImage;           // Image Buffer. Quicktime needs 3 bytes also for 8-bit png
                                    //   (biCompression==NO_COMPRESSION)?0:xDim*yDim*bytesPerPixel;
    int32_t  biXPelsPerMeter;       // Horizontal resolution in pixels per meter
    int32_t  biYPelsPerMeter;       // Vertical resolution in pixels per meter
    uint32_t biClrUsed;             // 256 (color table size; for 8-bit only)
    uint32_t biClrImportant;        // Specifies that the first x colors of the color table. Are important to the DIB.
};

struct RiffChunk
{
    uint32_t m_four_cc;
    uint32_t m_size;
};

struct RiffList
{
    uint32_t m_riff_or_list_cc;
    uint32_t m_size;
    uint32_t m_list_type_cc;
};

#pragma pack(pop)

}


typedef std::deque< std::pair<uint64_t, uint32_t> > frame_list;
typedef frame_list::iterator frame_iterator;

class nervana::MjpegInputStream
{
public:
    MjpegInputStream(){}
    virtual ~MjpegInputStream(){}
    virtual MjpegInputStream& read(char*, uint64_t) = 0;
    virtual MjpegInputStream& seekg(uint64_t) = 0;
    virtual uint64_t tellg() = 0;
    virtual bool isOpened() const = 0;
    virtual bool open(const std::string& filename) = 0;
    virtual void close() = 0;
    virtual operator bool() = 0;
};

class nervana::MjpegFileInputStream : public nervana::MjpegInputStream
{
public:
    MjpegFileInputStream();
    MjpegFileInputStream(const std::string& filename);
    ~MjpegFileInputStream();
    MjpegInputStream& read(char*, uint64_t) override;
    MjpegInputStream& seekg(uint64_t) override;
    uint64_t tellg() override;
    bool isOpened() const override;
    bool open(const std::string& filename) override;
    void close() override;
    operator bool() override;

private:
    bool            m_is_valid;
    std::ifstream   m_f;
};

class nervana::MjpegMemoryInputStream : public nervana::MjpegInputStream
{
public:
    MjpegMemoryInputStream();
    MjpegMemoryInputStream(char* data, size_t size);
    ~MjpegMemoryInputStream();
    MjpegInputStream& read(char*, uint64_t) override;
    MjpegInputStream& seekg(uint64_t) override;
    uint64_t tellg() override;
    bool isOpened() const override;
    bool open(const std::string& filename) override;
    void close() override;
    operator bool() override;

private:
    bool                        m_is_valid;
    nervana::memstream<char>    m_wrapper;
    std::istream                m_f;
};

nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::AviMainHeader& avih);
nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::AviStreamHeader& strh);
nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::BitmapInfoHeader& bmph);
nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::RiffList& riff_list);
nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::RiffChunk& riff_chunk);
nervana::MjpegInputStream& operator >> (nervana::MjpegInputStream& is, nervana::AviIndex& idx1);