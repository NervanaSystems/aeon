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
#include "log.hpp"

using namespace std;
using namespace cv;

uint64_t nervana::MotionJpegCapture::getFramePos() const
{
    if (m_is_first_frame)
        return 0;

    if (m_frame_iterator == m_mjpeg_frames.end())
        return m_mjpeg_frames.size();

    return m_frame_iterator - m_mjpeg_frames.begin() + 1;
}

bool nervana::MotionJpegCapture::setProperty(int property, double value)
{
    if (property == CV_CAP_PROP_POS_FRAMES)
    {
        if (int(value) == 0)
        {
            m_is_first_frame = true;
            m_frame_iterator = m_mjpeg_frames.end();
            return true;
        }
        else if (m_mjpeg_frames.size() > value)
        {
            m_frame_iterator = m_mjpeg_frames.begin() + int(value - 1);
            m_is_first_frame = false;
            return true;
        }
    }

    return false;
}

double nervana::MotionJpegCapture::getProperty(int property) const
{
    switch (property)
    {
    case CV_CAP_PROP_POS_FRAMES: return (double)getFramePos();
    case CV_CAP_PROP_POS_AVI_RATIO: return double(getFramePos()) / m_mjpeg_frames.size();
    case CV_CAP_PROP_FRAME_WIDTH: return (double)m_frame_width;
    case CV_CAP_PROP_FRAME_HEIGHT: return (double)m_frame_height;
    case CV_CAP_PROP_FPS: return m_fps;
    case CV_CAP_PROP_FOURCC: return (double)MJPG_CC;
    case CV_CAP_PROP_FRAME_COUNT: return (double)m_mjpeg_frames.size();
    case CV_CAP_PROP_FORMAT: return 0;
    default: return 0;
    }
}

std::vector<char> nervana::MotionJpegCapture::readFrame(frame_iterator it)
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

bool nervana::MotionJpegCapture::grabFrame()
{
    if (isOpened())
    {
        if (m_is_first_frame)
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

bool nervana::MotionJpegCapture::retrieveFrame(int, cv::Mat& output_frame)
{
    if (m_frame_iterator != m_mjpeg_frames.end())
    {
        std::vector<char> data = readFrame(m_frame_iterator);

        if (data.size())
        {
            output_frame = imdecode(data, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);
        }

        return true;
    }

    return false;
}

nervana::MotionJpegCapture::~MotionJpegCapture()
{
    close();
}

nervana::MotionJpegCapture::MotionJpegCapture(const string& filename)
{
    m_file_stream = make_shared<ifstream>(filename, ios::in | ios::binary);
    open();
}

nervana::MotionJpegCapture::MotionJpegCapture(char* buffer, size_t size)
{
    m_file_stream = make_shared<memory_stream>(buffer, size);
    open();
}

bool nervana::MotionJpegCapture::isOpened() const
{
    return m_mjpeg_frames.size() > 0;
}

void nervana::MotionJpegCapture::close()
{
    m_frame_iterator = m_mjpeg_frames.end();
}

bool nervana::MotionJpegCapture::open()
{
    m_frame_iterator = m_mjpeg_frames.end();
    m_is_first_frame = true;

    if (!parseRiff(*m_file_stream))
    {
        ERR << "Not a valid AVI file type" << endl;
        close();
    }

    return isOpened();
}

bool nervana::MotionJpegCapture::parseRiff(istream& in_str)
{
    bool result = false;
    while (in_str)
    {
        RiffList riff_list;

        in_str >> riff_list;

        if (in_str && riff_list.m_riff_or_list_cc == RIFF_CC &&
            ((riff_list.m_list_type_cc == AVI_CC) | (riff_list.m_list_type_cc == AVIX_CC)))
        {
            uint64_t next_riff = in_str.tellg();
            // RiffList::m_size includes fourCC field which we have already read
            next_riff += (riff_list.m_size - 4);

            AviMjpegStream mjpeg_video_stream;
            bool           is_parsed = mjpeg_video_stream.parseAvi(in_str, m_mjpeg_frames);
            result                   = result || is_parsed;

            if (is_parsed)
            {
                m_frame_width  = mjpeg_video_stream.getWidth();
                m_frame_height = mjpeg_video_stream.getHeight();
                m_fps          = mjpeg_video_stream.getFps();
            }

            in_str.seekg(next_riff);
        }
        else
        {
            break;
        }
    }
    in_str.clear();

    return result;
}
