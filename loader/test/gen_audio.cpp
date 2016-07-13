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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#ifdef HAVE_AV_CONFIG_H
#undef HAVE_AV_CONFIG_H
#endif

#include "gen_audio.hpp"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/common.h>
    #include <libavutil/opt.h>
}

// this stuff makes me cranky!
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc  avcodec_alloc_frame
#define av_frame_free   avcodec_free_frame
#endif

#define INBUF_SIZE 4096
#define AUDIO_INBUF_SIZE 20480
#define AUDIO_REFILL_THRESH 4096

using namespace std;
using nervana::pack_le;
using nervana::unpack_le;

wav_data::wav_data(char *buf, uint32_t bufsize)
{
    char *bptr = buf;
    if (bufsize < 36) {
        throw wavefile_exception("Header size is too small");
    }
    if (strncmp(bptr, "RIFF", 4) || strncmp(bptr+8, "WAVEfmt ", 8)) {
        throw wavefile_exception("Unsupported format");
    }

    uint32_t riff_chunk_size = unpack_le<uint32_t>(bptr + 4);

    if (riff_chunk_size > bufsize) {
        throw wavefile_exception("");
    }

    bptr += 16;

    uint32_t subchunk1_size  = unpack_le<uint32_t>(bptr);   bptr += sizeof(subchunk1_size);
    uint16_t audio_format    = unpack_le<uint16_t>(bptr);   bptr += sizeof(audio_format);
    uint16_t num_channels    = unpack_le<uint16_t>(bptr);   bptr += sizeof(num_channels);
    _sample_rate             = unpack_le<uint32_t>(bptr);   bptr += sizeof(_sample_rate);
    uint32_t byte_rate       = unpack_le<uint32_t>(bptr);   bptr += sizeof(byte_rate);
    uint32_t block_align     = unpack_le<uint16_t>(bptr);   bptr += sizeof(block_align);
    uint32_t bits_per_sample = unpack_le<uint16_t>(bptr);   bptr += sizeof(bits_per_sample);

    if (bits_per_sample != 16) {
        throw wavefile_exception("Ingested waveforms must be 16-bit PCM");
    }
    uint32_t fmt_chunk_read  = 16;

    if (audio_format != WAVE_FORMAT_PCM) {
        throw wavefile_exception("can read only PCM data");
    } else if (subchunk1_size < 16) {
        throw wavefile_exception("PCM format data not at least 16");
    }

    // Skip any subchunks between "fmt" and "data".
    while (strncmp(bptr, "data", 4) != 0) {
        uint32_t chunk_sz = unpack_le<uint32_t>(bptr + 4);
        if (chunk_sz != 4 && !strncmp(bptr, "fact", 4)) {
            throw wavefile_exception("Malformed fact chunk");
        }

        bptr += 4 + sizeof(chunk_sz) + chunk_sz; // chunk tag, chunk size, chunk
    }

    if (strncmp(bptr, "data", 4) != 0) {
        throw wavefile_exception("Got unexpected tag");
    }

    uint32_t data_chunk_size = unpack_le<uint32_t>(bptr + 4);

    bptr += 4 + sizeof(data_chunk_size);

    uint32_t num_samples = data_chunk_size / block_align;
    data.create(num_samples, num_channels, CV_16SC1);

    for (uint32_t n = 0; n < num_samples; ++n) {
        for (uint32_t c = 0; c < num_channels; ++c) {
            data.at<int16_t>(n, c) = unpack_le<int16_t>(bptr);
            bptr += sizeof(int16_t);
        }
    }
}


void wav_data::write_to_file(string filename)
{
    vector<char> stored_data(_file_size);
    write_header_alternate(stored_data);
    write_data(stored_data);

    _ofs.open(filename, ostream::binary);
    if(!_ofs) {
        throw std::runtime_error("couldn't write to file " + filename);
    }
    _ofs.write(&(stored_data[0]), _file_size);
    _ofs.close();
}

void wav_data::write_to_buffer(char *buf, uint32_t bufsize)
{
    if (bufsize < _file_size) {
        throw std::runtime_error("output buffer is too small " +
                                  std::to_string(bufsize) + " provided " +
                                  std::to_string(_file_size) + " needed.");
    }

    vector<char> stored_data(_file_size);
    write_header(stored_data);
    write_data(stored_data);
    memcpy(&(stored_data[0]), buf, _file_size);
}

void wav_data::write_header_alternate(vector<char> & buf)
{
    RiffMainHeader rh;
    rh.dwRiffCC      = nervana::FOURCC('R', 'I', 'F', 'F');
    rh.dwRiffLen     = sizeof(RiffMainHeader) + sizeof(FmtHeader) + sizeof(DataHeader) + nbytes();
    rh.dwWaveID      = nervana::FOURCC('W', 'A', 'V', 'E');

    FmtHeader fh;
    fh.dwFmtCC       = nervana::FOURCC('f', 'm', 't', ' ');
    fh.dwFmtLen      = sizeof(FmtHeader) - 2 * sizeof(uint32_t);
    fh.hwFmtTag      = WAVE_FORMAT_PCM;
    fh.hwChannels    = channels();
    fh.dwSampleRate  = sample_rate();
    fh.dwBytesPerSec = bps();
    fh.hwBlockAlign  = block_align();
    fh.hwBitDepth    = bit_depth();

    DataHeader dh;
    dh.dwDataCC      = nervana::FOURCC('d', 'a', 't', 'a');
    dh.dwDataLen     = nbytes();

    buf.resize(sizeof(rh) + sizeof(fh) + sizeof(dh));

    char *head_ptr = &(buf[0]);
    memcpy(head_ptr, &rh, sizeof(rh)); head_ptr += sizeof(rh);
    memcpy(head_ptr, &fh, sizeof(fh)); head_ptr += sizeof(fh);
    memcpy(head_ptr, &dh, sizeof(dh));

}

void wav_data::write_header(vector<char>& buf)
{
    int32_t fmtlen = 16;
    int16_t format_tag = 1;

    buf.resize(HEADER_SIZE);

    char *head_ptr = &(buf[0]);
    strncpy(head_ptr, "RIFF", 4);       head_ptr += 4;
    pack_le(head_ptr, _file_size);      head_ptr += sizeof(_file_size);
    strncpy(head_ptr, "WAVEfmt ", 8);   head_ptr += 8;
    pack_le(head_ptr, fmtlen);          head_ptr += sizeof(fmtlen);
    pack_le(head_ptr, format_tag);      head_ptr += sizeof(format_tag);
    pack_le(head_ptr, channels());      head_ptr += sizeof(int16_t);
    pack_le(head_ptr, sample_rate());   head_ptr += sizeof(int32_t);
    pack_le(head_ptr, bps());           head_ptr += sizeof(int32_t);
    pack_le(head_ptr, block_align());   head_ptr += sizeof(int16_t);
    pack_le(head_ptr, bit_depth());     head_ptr += sizeof(int16_t);

    strncpy(head_ptr, "data", 4);       head_ptr += 4;
    pack_le(head_ptr, nbytes());
}

void wav_data::write_data(vector<char>& buf)
{
    char *data_payload = &(buf[HEADER_SIZE]);
    for (int n = 0; n < data.rows; n++) {
        int16_t *ptr = data.ptr<int16_t>(n);
        for (int c = 0; c < data.cols; c++) {
            pack_le(data_payload, ptr[c], (n * data.cols + c) * sizeof(int16_t));
        }
    }
}



gen_audio::gen_audio() :
    r{42}
{
    // register all the codecs
    avcodec_register_all();

}

vector<unsigned char> gen_audio::render_target( int datumNumber ) {
    std::poisson_distribution<int> _word_count(3);
    std::uniform_int_distribution<int> _word(0,vocab.size()-1);
    vector<unsigned char> rc;

    int word_count = _word_count(r)+1;
    for(int i=0; i<word_count; i++) {
        int word = _word(r);
        string w = vocab[word];
        if( i > 0 ) rc.push_back(' ');
        rc.insert( rc.end(), w.begin(), w.end() );
    }
//    string result((char*)rc.data(),rc.size());
//    cout << "'" << result << "'\n";
    return rc;
}

vector<unsigned char> gen_audio::render_datum( int datumNumber ) {
    int frequency = ((datumNumber % 7) + 1) * 1000;
    return encode(frequency, 2000);
}

// NOTE: just descovered that this was mostly copy-pasted from here:
// https://ffmpeg.org/doxygen/trunk/encoding-example_8c-source.html

/*
 * Audio encoding example
 *
 * generate an mp2 encoded sin wave with `frequencyHz` and length `duration`
 * in milliseconds.
 */
vector<unsigned char> gen_audio::encode(float frequencyHz, int duration) {
    AVCodec *codec;
    AVCodecContext *c= nullptr;
    int frame_size, i, j, out_size, outbuf_size;
    short *samples;
    float t, tincr;
    uint8_t *outbuf;
    vector<unsigned char> rc;

    /* find the MP2 encoder */
    codec = avcodec_find_encoder(CODEC_ID_MP2);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);

    c->sample_fmt = c->codec->sample_fmts[0];

    /* put sample parameters */
    c->bit_rate = 64000;
    c->sample_rate = 44100;
    c->channels = 2;

    /* open it */
    if (avcodec_open2(c, codec, nullptr) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    /* the codec gives us the frame size, in samples */
    frame_size = c->frame_size;
    float frame_duration = (float)(c->frame_size) / (float)(c->sample_rate);
    int frames = ceil((float)duration / frame_duration / 1000.);
    samples = (short*)malloc(frame_size * 2 * c->channels);
    outbuf_size = 10000;
    outbuf = (uint8_t*)malloc(outbuf_size);

    /* encode a single tone sound */
    t = 0;
    tincr = 2 * M_PI * frequencyHz / c->sample_rate;
    for(i=0;i<frames;i++) {
        for(j=0;j<frame_size;j++) {
            samples[2*j] = (int)(sin(t) * 10000);
            samples[2*j+1] = samples[2*j];
            t += tincr;
        }
        /* encode the samples */
        out_size = avcodec_encode_audio(c, outbuf, outbuf_size, samples);
        for(int i=0; i<out_size; i++) { rc.push_back(outbuf[i]); }
    }
    free(outbuf);
    free(samples);

    avcodec_close(c);
    av_free(c);

    return rc;
}

/*
 * Audio encoding example
 *
 * generate an mp2 encoded sin wave with `frequencyHz` and length `duration`
 * in milliseconds.  write audio file to `filename`.
 */
void gen_audio::encode(const std::string& filename, float frequencyHz, int duration)
{
    FILE* f;
    f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename.c_str());
        exit(1);
    }

    vector<unsigned char> data = encode(frequencyHz,duration);
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
}

/*
 * Audio decoding.
 */
void gen_audio::decode(const std::string& outfilename, const std::string& filename)
{
    AVCodec *codec;
    AVCodecContext *c= NULL;
    int out_size, len;
    FILE *f, *outfile;
    uint8_t *outbuf;
    uint8_t inbuf[AUDIO_INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];
    AVPacket avpkt;

    av_init_packet(&avpkt);

    printf("Audio decoding\n");

    /* find the mpeg audio decoder */
    codec = avcodec_find_decoder(CODEC_ID_MP2);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c= avcodec_alloc_context3(codec);

    /* open it */
    if (avcodec_open2(c, codec, nullptr) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    outbuf = (uint8_t*)malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE);

    f = fopen(filename.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename.c_str());
        exit(1);
    }
    outfile = fopen(outfilename.c_str(), "wb");
    if (!outfile) {
        av_free(c);
        exit(1);
    }

    /* decode until eof */
    avpkt.data = inbuf;
    avpkt.size = fread(inbuf, 1, AUDIO_INBUF_SIZE, f);

    while (avpkt.size > 0) {
        out_size = AVCODEC_MAX_AUDIO_FRAME_SIZE;
        len = avcodec_decode_audio3(c, (short *)outbuf, &out_size, &avpkt);
        if (len < 0) {
            fprintf(stderr, "Error while decoding\n");
            exit(1);
        }
        if (out_size > 0) {
            /* if a frame has been decoded, output it */
            fwrite(outbuf, 1, out_size, outfile);
        }
        avpkt.size -= len;
        avpkt.data += len;
        if (avpkt.size < AUDIO_REFILL_THRESH) {
            /* Refill the input buffer, to avoid trying to decode
             * incomplete frames. Instead of this, one could also use
             * a parser, or use a proper container format through
             * libavformat. */
            memmove(inbuf, avpkt.data, avpkt.size);
            avpkt.data = inbuf;
            len = fread(avpkt.data + avpkt.size, 1,
                        AUDIO_INBUF_SIZE - avpkt.size, f);
            if (len > 0)
                avpkt.size += len;
        }
    }

    fclose(outfile);
    fclose(f);
    free(outbuf);

    avcodec_close(c);
    av_free(c);
}

vector<string> gen_audio::get_codec_list() {
    vector<string> rc;
    AVCodec* current_codec = av_codec_next(nullptr);
    while (current_codec != NULL)
    {
        if (!av_codec_is_encoder(current_codec))
        {
            current_codec = av_codec_next(current_codec);
            continue;
        }
        rc.push_back(string(current_codec->name));
        current_codec = av_codec_next(current_codec);
    }
    return rc;
}

