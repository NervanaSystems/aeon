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

#include "dataset.hpp"

class gen_audio : public dataset<gen_audio> {
public:
    void encode(const std::string& filename, float frequencyHz);
    void decode(const std::string& outfilename, const std::string& filename);

    static std::vector<std::string> get_codec_list();

private:
    std::vector<unsigned char> render_target( int datumNumber ) override;
    std::vector<unsigned char> render_datum( int datumNumber ) override;

    static const char* Encoder_GetNextCodecName();
    static const char* Encoder_GetFirstCodecName();
};


