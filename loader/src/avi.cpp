#include "avi.hpp"

using namespace std;

std::istream& operator>>(std::istream& is, nervana::AviMainHeader& avih)
{
    is.read((char*)(&avih), sizeof(nervana::AviMainHeader));
    return is;
}

std::istream& operator>>(std::istream& is, nervana::AviStreamHeader& strh)
{
    is.read((char*)(&strh), sizeof(nervana::AviStreamHeader));
    return is;
}

std::istream& operator>>(std::istream& is, nervana::BitmapInfoHeader& bmph)
{
    is.read((char*)(&bmph), sizeof(nervana::BitmapInfoHeader));
    return is;
}

std::istream& operator>>(std::istream& is, nervana::RiffList& riff_list)
{
    is.read((char*)(&riff_list), sizeof(riff_list));
    return is;
}

std::istream& operator>>(std::istream& is, nervana::RiffChunk& riff_chunk)
{
    is.read((char*)(&riff_chunk), sizeof(riff_chunk));
    return is;
}

std::istream& operator>>(std::istream& is, nervana::AviIndex& idx1)
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
    ss << (char)(fourcc & 255) << (char)((fourcc >> 8) & 255) << (char)((fourcc >> 16) & 255) << (char)((fourcc >> 24) & 255);
    return ss.str();
}

nervana::AviMjpegStream::AviMjpegStream()
    : m_stream_id(0)
    , m_movi_start(0)
    , m_movi_end(0)
    , m_width(0)
    , m_height(0)
    , m_fps(0)
    , m_is_indx_present(false)
{
}

size_t nervana::AviMjpegStream::getFramesCount()
{
    return m_frame_list.size();
}

frame_list& nervana::AviMjpegStream::getFrames()
{
    return m_frame_list;
}

uint32_t nervana::AviMjpegStream::getWidth()
{
    return m_width;
}

uint32_t nervana::AviMjpegStream::getHeight()
{
    return m_height;
}

double nervana::AviMjpegStream::getFps()
{
    return m_fps;
}

void nervana::AviMjpegStream::printError(istream& in_str, RiffList& list, uint32_t expected_fourcc)
{
    if (!in_str)
    {
        fprintf(stderr, "Unexpected end of file while searching for %s list\n", fourccToString(expected_fourcc).c_str());
    }
    else if (list.m_riff_or_list_cc != LIST_CC)
    {
        fprintf(stderr, "Unexpected element. Expected: %s. Got: %s.\n", fourccToString(LIST_CC).c_str(),
                fourccToString(list.m_riff_or_list_cc).c_str());
    }
    else
    {
        fprintf(stderr, "Unexpected list type. Expected: %s. Got: %s.\n", fourccToString(expected_fourcc).c_str(),
                fourccToString(list.m_list_type_cc).c_str());
    }
}

void nervana::AviMjpegStream::printError(istream& in_str, RiffChunk& chunk, uint32_t expected_fourcc)
{
    if (!in_str)
    {
        fprintf(stderr, "Unexpected end of file while searching for %s chunk\n", fourccToString(expected_fourcc).c_str());
    }
    else
    {
        fprintf(stderr, "Unexpected element. Expected: %s. Got: %s.\n", fourccToString(expected_fourcc).c_str(),
                fourccToString(chunk.m_four_cc).c_str());
    }
}

bool nervana::AviMjpegStream::parseMovi(istream&, frame_list&)
{
    // not implemented
    return true;
}

bool nervana::AviMjpegStream::parseInfo(istream&)
{
    // not implemented
    return true;
}

bool nervana::AviMjpegStream::parseIndex(istream& in_str, uint32_t index_size, frame_list& in_frame_list)
{
    uint64_t index_end = in_str.tellg();
    index_end += index_size;
    bool result = false;

    while (in_str && (in_str.tellg() < index_end))
    {
        AviIndex idx1;
        in_str >> idx1;

        if (idx1.ckid == m_stream_id)
        {
            uint64_t absolute_pos = m_movi_start + idx1.dwChunkOffset;

            if (absolute_pos < m_movi_end)
            {
                in_frame_list.push_back(std::make_pair(absolute_pos, idx1.dwChunkLength));
            }
            else
            {
                // unsupported case
                fprintf(stderr, "Frame offset points outside movi section.\n");
            }
        }

        result = true;
    }

    return result;
}

bool nervana::AviMjpegStream::parseStrl(istream& in_str, uint8_t stream_id)
{
    RiffChunk strh;
    in_str >> strh;

    if (in_str && strh.m_four_cc == STRH_CC)
    {
        uint64_t next_strl_list = in_str.tellg();
        next_strl_list += strh.m_size;

        AviStreamHeader strm_hdr;
        in_str >> strm_hdr;

        if (strm_hdr.fccType == VIDS_CC && strm_hdr.fccHandler == MJPG_CC)
        {
            uint8_t first_digit  = (stream_id / 10) + '0';
            uint8_t second_digit = (stream_id % 10) + '0';

            if (m_stream_id == 0)
            {
                m_stream_id = CV_FOURCC(first_digit, second_digit, 'd', 'c');
                m_fps       = double(strm_hdr.dwRate) / strm_hdr.dwScale;
            }
            else
            {
                // second mjpeg video stream found which is not supported
                fprintf(stderr, "More than one video stream found within AVI/AVIX list. Stream %c%cdc would be ignored\n",
                        first_digit, second_digit);
            }

            return true;
        }
    }

    return false;
}

void nervana::AviMjpegStream::skipJunk(RiffChunk& chunk, istream& in_str)
{
    if (chunk.m_four_cc == JUNK_CC)
    {
        in_str.seekg((uint32_t)in_str.tellg() + chunk.m_size);
        in_str >> chunk;
    }
}

void nervana::AviMjpegStream::skipJunk(RiffList& list, istream& in_str)
{
    if (list.m_riff_or_list_cc == JUNK_CC)
    {
        // JUNK chunk is 4 bytes less than LIST
        in_str.seekg((uint32_t)in_str.tellg() + list.m_size - 4);
        in_str >> list;
    }
}

bool nervana::AviMjpegStream::parseHdrlList(istream& in_str)
{
    bool result = false;

    RiffChunk avih;
    in_str >> avih;

    if (in_str && avih.m_four_cc == AVIH_CC)
    {
        uint64_t next_strl_list = in_str.tellg();
        next_strl_list += avih.m_size;

        AviMainHeader avi_hdr;
        in_str >> avi_hdr;

        if (in_str)
        {
            m_is_indx_present          = ((avi_hdr.dwFlags & 0x10) != 0);
            uint32_t number_of_streams = avi_hdr.dwStreams;
            m_width                    = avi_hdr.dwWidth;
            m_height                   = avi_hdr.dwHeight;

            // the number of strl lists must be equal to number of streams specified in main avi header
            for (uint32_t i = 0; i < number_of_streams; ++i)
            {
                in_str.seekg(next_strl_list);
                RiffList strl_list;
                in_str >> strl_list;

                if (in_str && strl_list.m_riff_or_list_cc == LIST_CC && strl_list.m_list_type_cc == STRL_CC)
                {
                    next_strl_list = in_str.tellg();
                    // RiffList::m_size includes fourCC field which we have already read
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

bool nervana::AviMjpegStream::parseAviWithFrameList(istream& in_str, frame_list& in_frame_list)
{
    RiffList hdrl_list;
    in_str >> hdrl_list;

    if (in_str && hdrl_list.m_riff_or_list_cc == LIST_CC && hdrl_list.m_list_type_cc == HDRL_CC)
    {
        uint64_t next_list = in_str.tellg();
        // RiffList::m_size includes fourCC field which we have already read
        next_list += (hdrl_list.m_size - 4);
        // parseHdrlList sets m_is_indx_present flag which would be used later
        if (parseHdrlList(in_str))
        {
            in_str.seekg(next_list);

            RiffList some_list;
            in_str >> some_list;

            // an optional section INFO
            if (in_str && some_list.m_riff_or_list_cc == LIST_CC && some_list.m_list_type_cc == INFO_CC)
            {
                next_list = in_str.tellg();
                // RiffList::m_size includes fourCC field which we have already read
                next_list += (some_list.m_size - 4);
                parseInfo(in_str);

                in_str.seekg(next_list);
                in_str >> some_list;
            }

            // an optional section JUNK
            skipJunk(some_list, in_str);

            // we are expecting to find here movi list. Must present in avi
            if (in_str && some_list.m_riff_or_list_cc == LIST_CC && some_list.m_list_type_cc == MOVI_CC)
            {
                bool is_index_found = false;

                m_movi_start = in_str.tellg();
                m_movi_start -= 4;

                m_movi_end = m_movi_start + some_list.m_size;
                // if m_is_indx_present is set to true we should find index
                if (m_is_indx_present)
                {
                    // we are expecting to find index section after movi list
                    uint32_t indx_pos = (uint32_t)m_movi_start + 4;
                    indx_pos += (some_list.m_size - 4);
                    in_str.seekg(indx_pos);

                    RiffChunk index_chunk;
                    in_str >> index_chunk;

                    if (in_str && index_chunk.m_four_cc == IDX1_CC)
                    {
                        is_index_found = parseIndex(in_str, index_chunk.m_size, in_frame_list);
                        // we are not going anywhere else
                    }
                    else
                    {
                        printError(in_str, index_chunk, IDX1_CC);
                    }
                }
                // index not present or we were not able to find it
                // parsing movi list
                if (!is_index_found)
                {
                    // not implemented
                    parseMovi(in_str, in_frame_list);

                    fprintf(stderr, "Failed to parse avi: index was not found\n");
                    // we are not going anywhere else
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

bool nervana::AviMjpegStream::parseAvi(istream& in_str, frame_list& in_frame_list)
{
    return parseAviWithFrameList(in_str, in_frame_list);
}

bool nervana::AviMjpegStream::parseAvi(istream& in_str)
{
    return parseAviWithFrameList(in_str, m_frame_list);
}
