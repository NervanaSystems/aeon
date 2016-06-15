/*
 * Copyright (c) 2001 Fabrice Bellard
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <string>
#include <vector>

class avgen {
public:
    /*
     * Audio encoding example
     */
    static void audio_encode(const std::string& filename, float frequencyHz);

    /*
     * Audio decoding.
     */
    static void audio_decode(const std::string& outfilename, const std::string& filename);

    /*
     * Video encoding example
     */
    static void video_encode(const std::string& filename);

    /*
     * Video decoding example
     */
    static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                         char *filename);

    static void video_decode(const std::string& outfilename, const std::string& filename);

    static std::vector<std::string> get_codec_list();

private:
    static const char* Encoder_GetNextCodecName();
    static const char* Encoder_GetFirstCodecName();
};


