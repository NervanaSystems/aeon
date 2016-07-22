/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "cap_mjpeg_decoder.hpp"
#include "avi.hpp"

using namespace std;
using namespace cv;

namespace nervana
{

#define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))

int CV_FOURCC(char c1, char c2, char c3, char c4)
{
    return CV_FOURCC_MACRO(c1, c2, c3, c4);
}

const uint32_t RIFF_CC = CV_FOURCC('R','I','F','F');
const uint32_t LIST_CC = CV_FOURCC('L','I','S','T');
const uint32_t HDRL_CC = CV_FOURCC('h','d','r','l');
const uint32_t AVIH_CC = CV_FOURCC('a','v','i','h');
const uint32_t STRL_CC = CV_FOURCC('s','t','r','l');
const uint32_t STRH_CC = CV_FOURCC('s','t','r','h');
const uint32_t VIDS_CC = CV_FOURCC('v','i','d','s');
const uint32_t MJPG_CC = CV_FOURCC('M','J','P','G');
const uint32_t MOVI_CC = CV_FOURCC('m','o','v','i');
const uint32_t IDX1_CC = CV_FOURCC('i','d','x','1');
const uint32_t AVI_CC  = CV_FOURCC('A','V','I',' ');
const uint32_t AVIX_CC = CV_FOURCC('A','V','I','X');
const uint32_t JUNK_CC = CV_FOURCC('J','U','N','K');
const uint32_t INFO_CC = CV_FOURCC('I','N','F','O');

string fourccToString(uint32_t fourcc)
{
    stringstream ss;
    ss <<  (fourcc & 255) << ((fourcc >> 8) & 255) << ((fourcc >> 16) & 255) << ((fourcc >> 24) & 255);
    return ss.str();
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

//Represents single MJPEG video stream within single AVI/AVIX entry
//Multiple video streams within single AVI/AVIX entry are not supported
//ODML index is not supported
class AviMjpegStream
{
public:
    AviMjpegStream();
    //stores founded frames in m_frame_list which can be accessed via getFrames
    bool parseAvi(MjpegInputStream& in_str);
    //stores founded frames in in_frame_list. getFrames() would return empty list
    bool parseAvi(MjpegInputStream& in_str, frame_list& in_frame_list);
    size_t getFramesCount();
    frame_list& getFrames();
    uint32_t getWidth();
    uint32_t getHeight();
    double getFps();

protected:

    bool parseAviWithFrameList(MjpegInputStream& in_str, frame_list& in_frame_list);
    void skipJunk(RiffChunk& chunk, MjpegInputStream& in_str);
    void skipJunk(RiffList& list, MjpegInputStream& in_str);
    bool parseHdrlList(MjpegInputStream& in_str);
    bool parseIndex(MjpegInputStream& in_str, uint32_t index_size, frame_list& in_frame_list);
    bool parseMovi(MjpegInputStream& in_str, frame_list& in_frame_list);
    bool parseStrl(MjpegInputStream& in_str, uint8_t stream_id);
    bool parseInfo(MjpegInputStream& in_str);
    void printError(MjpegInputStream& in_str, RiffList& list, uint32_t expected_fourcc);
    void printError(MjpegInputStream& in_str, RiffChunk& chunk, uint32_t expected_fourcc);

    uint32_t   m_stream_id;
    uint64_t   m_movi_start;
    uint64_t   m_movi_end;
    frame_list m_frame_list;
    uint32_t   m_width;
    uint32_t   m_height;
    double     m_fps;
    bool       m_is_indx_present;
};

AviMjpegStream::AviMjpegStream(): m_stream_id(0), m_movi_start(0), m_movi_end(0), m_width(0), m_height(0), m_fps(0), m_is_indx_present(false)
{
}

size_t AviMjpegStream::getFramesCount()
{
    return m_frame_list.size();
}

frame_list& AviMjpegStream::getFrames()
{
    return m_frame_list;
}

uint32_t AviMjpegStream::getWidth()
{
    return m_width;
}

uint32_t AviMjpegStream::getHeight()
{
    return m_height;
}

double AviMjpegStream::getFps()
{
    return m_fps;
}

void AviMjpegStream::printError(MjpegInputStream& in_str, RiffList& list, uint32_t expected_fourcc)
{
    if(!in_str)
    {
        fprintf(stderr, "Unexpected end of file while searching for %s list\n", fourccToString(expected_fourcc).c_str());
    }
    else if(list.m_riff_or_list_cc != LIST_CC)
    {
        fprintf(stderr, "Unexpected element. Expected: %s. Got: %s.\n", fourccToString(LIST_CC).c_str(), fourccToString(list.m_riff_or_list_cc).c_str());
    }
    else
    {
        fprintf(stderr, "Unexpected list type. Expected: %s. Got: %s.\n", fourccToString(expected_fourcc).c_str(), fourccToString(list.m_list_type_cc).c_str());
    }
}

void AviMjpegStream::printError(MjpegInputStream& in_str, RiffChunk& chunk, uint32_t expected_fourcc)
{
    if(!in_str)
    {
        fprintf(stderr, "Unexpected end of file while searching for %s chunk\n", fourccToString(expected_fourcc).c_str());
    }
    else
    {
        fprintf(stderr, "Unexpected element. Expected: %s. Got: %s.\n", fourccToString(expected_fourcc).c_str(), fourccToString(chunk.m_four_cc).c_str());
    }
}


bool AviMjpegStream::parseMovi(MjpegInputStream&, frame_list&)
{
    //not implemented
    return true;
}

bool AviMjpegStream::parseInfo(MjpegInputStream&)
{
    //not implemented
    return true;
}

bool AviMjpegStream::parseIndex(MjpegInputStream& in_str, uint32_t index_size, frame_list& in_frame_list)
{
    uint64_t index_end = in_str.tellg();
    index_end += index_size;
    bool result = false;

    while(in_str && (in_str.tellg() < index_end))
    {
        AviIndex idx1;
        in_str >> idx1;

        if(idx1.ckid == m_stream_id)
        {
            uint64_t absolute_pos = m_movi_start + idx1.dwChunkOffset;

            if(absolute_pos < m_movi_end)
            {
                in_frame_list.push_back(std::make_pair(absolute_pos, idx1.dwChunkLength));
            }
            else
            {
                //unsupported case
                fprintf(stderr, "Frame offset points outside movi section.\n");
            }
        }

        result = true;
    }

    return result;
}

bool AviMjpegStream::parseStrl(MjpegInputStream& in_str, uint8_t stream_id)
{
    RiffChunk strh;
    in_str >> strh;

    if(in_str && strh.m_four_cc == STRH_CC)
    {
        uint64_t next_strl_list = in_str.tellg();
        next_strl_list += strh.m_size;

        AviStreamHeader strm_hdr;
        in_str >> strm_hdr;

        if(strm_hdr.fccType == VIDS_CC && strm_hdr.fccHandler == MJPG_CC)
        {
            uint8_t first_digit = (stream_id/10) + '0';
            uint8_t second_digit = (stream_id%10) + '0';

            if(m_stream_id == 0)
            {
                m_stream_id = CV_FOURCC(first_digit, second_digit, 'd', 'c');
                m_fps = double(strm_hdr.dwRate)/strm_hdr.dwScale;
            }
            else
            {
                //second mjpeg video stream found which is not supported
                fprintf(stderr, "More than one video stream found within AVI/AVIX list. Stream %c%cdc would be ignored\n", first_digit, second_digit);
            }

            return true;
        }
    }

    return false;
}

void AviMjpegStream::skipJunk(RiffChunk& chunk, MjpegInputStream& in_str)
{
    if(chunk.m_four_cc == JUNK_CC)
    {
        in_str.seekg(in_str.tellg() + chunk.m_size);
        in_str >> chunk;
    }
}

void AviMjpegStream::skipJunk(RiffList& list, MjpegInputStream& in_str)
{
    if(list.m_riff_or_list_cc == JUNK_CC)
    {
        //JUNK chunk is 4 bytes less than LIST
        in_str.seekg(in_str.tellg() + list.m_size - 4);
        in_str >> list;
    }
}

bool AviMjpegStream::parseHdrlList(MjpegInputStream& in_str)
{
    bool result = false;

    RiffChunk avih;
    in_str >> avih;

    if(in_str && avih.m_four_cc == AVIH_CC)
    {
        uint64_t next_strl_list = in_str.tellg();
        next_strl_list += avih.m_size;

        AviMainHeader avi_hdr;
        in_str >> avi_hdr;

        if(in_str)
        {
            m_is_indx_present = ((avi_hdr.dwFlags & 0x10) != 0);
            DWORD number_of_streams = avi_hdr.dwStreams;
            m_width = avi_hdr.dwWidth;
            m_height = avi_hdr.dwHeight;

            //the number of strl lists must be equal to number of streams specified in main avi header
            for(DWORD i = 0; i < number_of_streams; ++i)
            {
                in_str.seekg(next_strl_list);
                RiffList strl_list;
                in_str >> strl_list;

                if( in_str && strl_list.m_riff_or_list_cc == LIST_CC && strl_list.m_list_type_cc == STRL_CC )
                {
                    next_strl_list = in_str.tellg();
                    //RiffList::m_size includes fourCC field which we have already read
                    next_strl_list += (strl_list.m_size - 4);

                    result = parseStrl(in_str, (uint8_t)i);
                }
                else
                {
                    printError(in_str, strl_list, STRL_CC);
                }
            }
        }
    }
    else
    {
        printError(in_str, avih, AVIH_CC);
    }

    return result;
}

bool AviMjpegStream::parseAviWithFrameList(MjpegInputStream& in_str, frame_list& in_frame_list)
{
    RiffList hdrl_list;
    in_str >> hdrl_list;

    if( in_str && hdrl_list.m_riff_or_list_cc == LIST_CC && hdrl_list.m_list_type_cc == HDRL_CC )
    {
        uint64_t next_list = in_str.tellg();
        //RiffList::m_size includes fourCC field which we have already read
        next_list += (hdrl_list.m_size - 4);
        //parseHdrlList sets m_is_indx_present flag which would be used later
        if(parseHdrlList(in_str))
        {
            in_str.seekg(next_list);

            RiffList some_list;
            in_str >> some_list;

            //an optional section INFO
            if(in_str && some_list.m_riff_or_list_cc == LIST_CC && some_list.m_list_type_cc == INFO_CC)
            {
                next_list = in_str.tellg();
                //RiffList::m_size includes fourCC field which we have already read
                next_list += (some_list.m_size - 4);
                parseInfo(in_str);

                in_str.seekg(next_list);
                in_str >> some_list;
            }

            //an optional section JUNK
            skipJunk(some_list, in_str);

            //we are expecting to find here movi list. Must present in avi
            if(in_str && some_list.m_riff_or_list_cc == LIST_CC && some_list.m_list_type_cc == MOVI_CC)
            {
                bool is_index_found = false;

                m_movi_start = in_str.tellg();
                m_movi_start -= 4;

                m_movi_end = m_movi_start + some_list.m_size;
                //if m_is_indx_present is set to true we should find index
                if(m_is_indx_present)
                {
                    //we are expecting to find index section after movi list
                    uint32_t indx_pos = (uint32_t)m_movi_start + 4;
                    indx_pos += (some_list.m_size - 4);
                    in_str.seekg(indx_pos);

                    RiffChunk index_chunk;
                    in_str >> index_chunk;

                    if(in_str && index_chunk.m_four_cc == IDX1_CC)
                    {
                        is_index_found = parseIndex(in_str, index_chunk.m_size, in_frame_list);
                        //we are not going anywhere else
                    }
                    else
                    {
                        printError(in_str, index_chunk, IDX1_CC);
                    }
                }
                //index not present or we were not able to find it
                //parsing movi list
                if(!is_index_found)
                {
                    //not implemented
                    parseMovi(in_str, in_frame_list);

                    fprintf(stderr, "Failed to parse avi: index was not found\n");
                    //we are not going anywhere else
                }
            }
            else
            {
                printError(in_str, some_list, MOVI_CC);
            }
        }
    }
    else
    {
        printError(in_str, hdrl_list, HDRL_CC);
    }

    return in_frame_list.size() > 0;
}

bool AviMjpegStream::parseAvi(MjpegInputStream& in_str, frame_list& in_frame_list)
{
    return parseAviWithFrameList(in_str, in_frame_list);
}

bool AviMjpegStream::parseAvi(MjpegInputStream& in_str)
{
    return parseAviWithFrameList(in_str, m_frame_list);
}

uint64_t MotionJpegCapture::getFramePos() const
{
    if(m_is_first_frame)
        return 0;

    if(m_frame_iterator == m_mjpeg_frames.end())
        return m_mjpeg_frames.size();

    return m_frame_iterator - m_mjpeg_frames.begin() + 1;
}

bool MotionJpegCapture::setProperty(int property, double value)
{
    if(property == CV_CAP_PROP_POS_FRAMES)
    {
        if(int(value) == 0)
        {
            m_is_first_frame = true;
            m_frame_iterator = m_mjpeg_frames.end();
            return true;
        }
        else if(m_mjpeg_frames.size() > value)
        {
            m_frame_iterator = m_mjpeg_frames.begin() + int(value - 1);
            m_is_first_frame = false;
            return true;
        }
    }

    return false;
}

double MotionJpegCapture::getProperty(int property) const
{
    switch(property)
    {
        case CV_CAP_PROP_POS_FRAMES:
            return (double)getFramePos();
        case CV_CAP_PROP_POS_AVI_RATIO:
            return double(getFramePos())/m_mjpeg_frames.size();
        case CV_CAP_PROP_FRAME_WIDTH:
            return (double)m_frame_width;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return (double)m_frame_height;
        case CV_CAP_PROP_FPS:
            return m_fps;
        case CV_CAP_PROP_FOURCC:
            return (double)CV_FOURCC('M','J','P','G');
        case CV_CAP_PROP_FRAME_COUNT:
            return (double)m_mjpeg_frames.size();
        case CV_CAP_PROP_FORMAT:
            return 0;
        default:
            return 0;
    }
}

std::vector<char> MotionJpegCapture::readFrame(frame_iterator it)
{
    m_file_stream->seekg(it->first);

    RiffChunk chunk;
    *m_file_stream >> chunk;

    std::vector<char> result;

    result.reserve(chunk.m_size);
    result.resize(chunk.m_size);

    m_file_stream->read(&(result[0]), chunk.m_size); // result.data() failed with MSVS2008

    return result;
}

bool MotionJpegCapture::grabFrame()
{
    if(isOpened())
    {
        if(m_is_first_frame)
        {
            m_is_first_frame = false;
            m_frame_iterator = m_mjpeg_frames.begin();
        }
        else
        {
            ++m_frame_iterator;
        }
    }

    return m_frame_iterator != m_mjpeg_frames.end();
}

bool MotionJpegCapture::retrieveFrame(int, OutputArray output_frame)
{
    if(m_frame_iterator != m_mjpeg_frames.end())
    {
        std::vector<char> data = readFrame(m_frame_iterator);

        if(data.size())
        {
            m_current_frame = imdecode(data, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);
        }

        m_current_frame.copyTo(output_frame);

        return true;
    }

    return false;
}

MotionJpegCapture::~MotionJpegCapture()
{
    close();
}

MotionJpegCapture::MotionJpegCapture(const string& filename)
{
    m_file_stream = make_shared<MjpegFileInputStream>(filename);
    open();
}

MotionJpegCapture::MotionJpegCapture(char* buffer, size_t size)
{
    m_file_stream = make_shared<MjpegMemoryInputStream>(buffer, size);
    open();
}


bool MotionJpegCapture::isOpened() const
{
    return m_mjpeg_frames.size() > 0;
}

void MotionJpegCapture::close()
{
    m_file_stream->close();
    m_frame_iterator = m_mjpeg_frames.end();
}

bool MotionJpegCapture::open()
{
//    close();

//    m_file_stream->open(filename);

    m_frame_iterator = m_mjpeg_frames.end();
    m_is_first_frame = true;

    if(!parseRiff(*m_file_stream))
    {
        close();
    }

    return isOpened();
}


bool MotionJpegCapture::parseRiff(MjpegInputStream& in_str)
{
    bool result = false;
    while(in_str)
    {
        RiffList riff_list;

        in_str >> riff_list;

        if( in_str && riff_list.m_riff_or_list_cc == RIFF_CC &&
            ((riff_list.m_list_type_cc == AVI_CC) | (riff_list.m_list_type_cc == AVIX_CC)) )
        {
            uint64_t next_riff = in_str.tellg();
            //RiffList::m_size includes fourCC field which we have already read
            next_riff += (riff_list.m_size - 4);

            AviMjpegStream mjpeg_video_stream;
            bool is_parsed = mjpeg_video_stream.parseAvi(in_str, m_mjpeg_frames);
            result = result || is_parsed;

            if(is_parsed)
            {
                m_frame_width = mjpeg_video_stream.getWidth();
                m_frame_height = mjpeg_video_stream.getHeight();
                m_fps = mjpeg_video_stream.getFps();
            }

            in_str.seekg(next_riff);
        }
        else
        {
            break;
        }
    }

    return result;
}

Ptr<MotionJpegCapture> createMotionJpegCapture(const string& filename)
{
    Ptr<MotionJpegCapture> mjdecoder(new MotionJpegCapture(filename));
    if( mjdecoder->isOpened() )
        return mjdecoder;
    return Ptr<MotionJpegCapture>();
}

}