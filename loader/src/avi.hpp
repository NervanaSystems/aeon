#pragma once

#include <deque>
#include <string>
#include <fstream>
#include <sstream>
#include <cinttypes>

#include "util.hpp"

namespace nervana
{
    struct AviMainHeader;
    struct AviStreamHeader;
    struct AviIndex;
    struct BitmapInfoHeader;
    struct RiffChunk;
    struct RiffList;
    class AviMjpegStream;
}

#pragma pack(push, 1)
struct nervana::AviMainHeader
{
    uint32_t dwMicroSecPerFrame; //  The period between video frames
    uint32_t dwMaxBytesPerSec;   //  Maximum data rate of the file
    uint32_t dwReserved1;        // 0
    uint32_t dwFlags;            //  0x10 AVIF_HASINDEX: The AVI file has an idx1 chunk containing an index at the end of the file.
    uint32_t dwTotalFrames;      // Field of the main header specifies the total number of frames of data in file.
    uint32_t dwInitialFrames;    // Is used for interleaved files
    uint32_t dwStreams;          // Specifies the number of streams in the file.
    uint32_t dwSuggestedBufferSize; // Field specifies the suggested buffer size forreading the file
    uint32_t dwWidth;               // Fields specify the width of the AVIfile in pixels.
    uint32_t dwHeight;              // Fields specify the height of the AVIfile in pixels.
    uint32_t dwReserved[4];         // 0, 0, 0, 0
};

struct nervana::AviStreamHeader
{
    uint32_t fccType;               // 'vids', 'auds', 'txts'...
    uint32_t fccHandler;            // "cvid", "DIB "
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
    struct
    {
        int16_t left;
        int16_t top;
        int16_t right;
        int16_t bottom;
    } rcFrame; // If stream has a different size than dwWidth*dwHeight(unused)
};

struct nervana::AviIndex
{
    uint32_t ckid;
    uint32_t dwFlags;
    uint32_t dwChunkOffset;
    uint32_t dwChunkLength;
};

struct nervana::BitmapInfoHeader
{
    uint32_t biSize;          // Write header size of BITMAPINFO header structure
    int32_t  biWidth;         // width in pixels
    int32_t  biHeight;        // heigth in pixels
    uint16_t biPlanes;        // Number of color planes in which the data is stored
    uint16_t biBitCount;      // Number of bits per pixel
    uint32_t biCompression;   // Type of compression used (uncompressed: NO_COMPRESSION=0)
    uint32_t biSizeImage;     // Image Buffer. Quicktime needs 3 bytes also for 8-bit png
                              //   (biCompression==NO_COMPRESSION)?0:xDim*yDim*bytesPerPixel;
    int32_t  biXPelsPerMeter; // Horizontal resolution in pixels per meter
    int32_t  biYPelsPerMeter; // Vertical resolution in pixels per meter
    uint32_t biClrUsed;       // 256 (color table size; for 8-bit only)
    uint32_t biClrImportant;  // Specifies that the first x colors of the color table. Are important to the DIB.
};

struct nervana::RiffChunk
{
    uint32_t m_four_cc;
    uint32_t m_size;
};

struct nervana::RiffList
{
    uint32_t m_riff_or_list_cc;
    uint32_t m_size;
    uint32_t m_list_type_cc;
};
#pragma pack(pop)

typedef std::deque<std::pair<uint64_t, uint32_t>> frame_list;
typedef frame_list::iterator frame_iterator;

std::istream& operator>>(std::istream& is, nervana::AviMainHeader& avih);
std::istream& operator>>(std::istream& is, nervana::AviStreamHeader& strh);
std::istream& operator>>(std::istream& is, nervana::BitmapInfoHeader& bmph);
std::istream& operator>>(std::istream& is, nervana::RiffList& riff_list);
std::istream& operator>>(std::istream& is, nervana::RiffChunk& riff_chunk);
std::istream& operator>>(std::istream& is, nervana::AviIndex& idx1);

namespace nervana
{
#define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1)&255) + (((c2)&255) << 8) + (((c3)&255) << 16) + (((c4)&255) << 24))

    int CV_FOURCC(char c1, char c2, char c3, char c4);

    const uint32_t RIFF_CC = CV_FOURCC('R', 'I', 'F', 'F');
    const uint32_t LIST_CC = CV_FOURCC('L', 'I', 'S', 'T');
    const uint32_t HDRL_CC = CV_FOURCC('h', 'd', 'r', 'l');
    const uint32_t AVIH_CC = CV_FOURCC('a', 'v', 'i', 'h');
    const uint32_t STRL_CC = CV_FOURCC('s', 't', 'r', 'l');
    const uint32_t STRH_CC = CV_FOURCC('s', 't', 'r', 'h');
    const uint32_t VIDS_CC = CV_FOURCC('v', 'i', 'd', 's');
    const uint32_t MJPG_CC = CV_FOURCC('M', 'J', 'P', 'G');
    const uint32_t MOVI_CC = CV_FOURCC('m', 'o', 'v', 'i');
    const uint32_t IDX1_CC = CV_FOURCC('i', 'd', 'x', '1');
    const uint32_t AVI_CC  = CV_FOURCC('A', 'V', 'I', ' ');
    const uint32_t AVIX_CC = CV_FOURCC('A', 'V', 'I', 'X');
    const uint32_t JUNK_CC = CV_FOURCC('J', 'U', 'N', 'K');
    const uint32_t INFO_CC = CV_FOURCC('I', 'N', 'F', 'O');

    std::string fourccToString(uint32_t fourcc);
}

/*
AVI struct:

RIFF ('AVI '
      LIST ('hdrl'
            'avih'(<Main AVI Header>)
            LIST ('strl'
                  'strh'(<Stream header>)
                  'strf'(<Stream format>)
                  [ 'strd'(<Additional header data>) ]
                  [ 'strn'(<Stream name>) ]
                  [ 'indx'(<Odml index data>) ]
                  ...
                 )
            [LIST ('strl' ...)]
            [LIST ('strl' ...)]
            ...
            [LIST ('odml'
                  'dmlh'(<ODML header data>)
                  ...
                 )
            ]
            ...
           )
      [LIST ('INFO' ...)]
      [JUNK]
      LIST ('movi'
            {{xxdb|xxdc|xxpc|xxwb}(<Data>) | LIST ('rec '
                              {xxdb|xxdc|xxpc|xxwb}(<Data>)
                              {xxdb|xxdc|xxpc|xxwb}(<Data>)
                              ...
                             )
               ...
            }
            ...
           )
      ['idx1' (<AVI Index>) ]
     )

     {xxdb|xxdc|xxpc|xxwb}
     xx - stream number: 00, 01, 02, ...
     db - uncompressed video frame
     dc - commpressed video frame
     pc - palette change
     wb - audio frame

     JUNK section may pad any data section and must be ignored
*/

// Represents single MJPEG video stream within single AVI/AVIX entry
// Multiple video streams within single AVI/AVIX entry are not supported
// ODML index is not supported
class nervana::AviMjpegStream
{
public:
    AviMjpegStream();
    // stores founded frames in m_frame_list which can be accessed via getFrames
    bool parseAvi(std::istream& in_str);
    // stores founded frames in in_frame_list. getFrames() would return empty list
    bool parseAvi(std::istream& in_str, frame_list& in_frame_list);
    size_t      getFramesCount();
    frame_list& getFrames();
    uint32_t    getWidth();
    uint32_t    getHeight();
    double      getFps();

protected:
    bool parseAviWithFrameList(std::istream& in_str, frame_list& in_frame_list);
    void skipJunk(RiffChunk& chunk, std::istream& in_str);
    void skipJunk(RiffList& list, std::istream& in_str);
    bool parseHdrlList(std::istream& in_str);
    bool parseIndex(std::istream& in_str, uint32_t index_size, frame_list& in_frame_list);
    bool parseMovi(std::istream& in_str, frame_list& in_frame_list);
    bool parseStrl(std::istream& in_str, uint8_t stream_id);
    bool parseInfo(std::istream& in_str);
    void printError(std::istream& in_str, RiffList& list, uint32_t expected_fourcc);
    void printError(std::istream& in_str, RiffChunk& chunk, uint32_t expected_fourcc);

    uint32_t   m_stream_id;
    uint64_t   m_movi_start;
    uint64_t   m_movi_end;
    frame_list m_frame_list;
    uint32_t   m_width;
    uint32_t   m_height;
    double     m_fps;
    bool       m_is_indx_present;
};
