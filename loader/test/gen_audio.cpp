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

    wav_assert(bufsize >= HEADER_SIZE, "Header size is too small");

    RiffMainHeader rh;
    FmtHeader fh;

    memcpy(&rh, bptr, sizeof(rh)); bptr += sizeof(rh);
    memcpy(&fh, bptr, sizeof(fh)); bptr += sizeof(fh);

    wav_assert(rh.dwRiffCC == nervana::FOURCC('R', 'I', 'F', 'F'), "Unsupported format");
    wav_assert(rh.dwWaveID == nervana::FOURCC('W', 'A', 'V', 'E'), "Unsupported format");
    wav_assert(bufsize >= rh.dwRiffLen, "Buffer not large enough for indicated file size");

    wav_assert(fh.hwFmtTag == WAVE_FORMAT_PCM, "can read only PCM data");
    wav_assert(fh.hwChannels == 16, "Ingested waveforms must be 16-bit PCM");
    wav_assert(fh.dwFmtLen >= 16, "PCM format data must be at least 16 bytes");

    // Skip any subchunks between "fmt" and "data".
    while (strncmp(bptr, "data", 4) != 0) {
        uint32_t chunk_sz = unpack_le<uint32_t>(bptr + 4);
        if (chunk_sz != 4 && !strncmp(bptr, "fact", 4)) {
            throw wavefile_exception("Malformed fact chunk");
        }
        bptr += 4 + sizeof(chunk_sz) + chunk_sz; // chunk tag, chunk size, chunk
    }

    wav_assert(strncmp(bptr, "data", 4) == 0, "Expected data tag not found");

    DataHeader dh;
    memcpy(&dh, bptr, sizeof(dh)); bptr += sizeof(dh);

    uint32_t num_samples = dh.dwDataLen / fh.hwBlockAlign;
    data.create(num_samples, fh.hwChannels, CV_16SC1);
    _sample_rate = fh.dwSampleRate;

    for (uint32_t n = 0; n < data.rows; ++n) {
        for (uint32_t c = 0; c < data.cols; ++c) {
            data.at<int16_t>(n, c) = unpack_le<int16_t>(bptr);
            bptr += sizeof(int16_t);
        }
    }
}
void wav_data::dump(std::ostream & ostr)
{
    ostr << "sample_rate " << _sample_rate << "\n";
    ostr << "channels x samples " << data.size() << "\n";
    ostr << "bit_depth " << (data.elemSize() * 8) << "\n";
    ostr << "nbytes " << nbytes() << "\n";
}

void wav_data::write_to_file(string filename)
{
    uint32_t totsize = HEADER_SIZE + nbytes();
    char* buf = new char[totsize];

    write_to_buffer(buf, totsize);

    std::ofstream ofs;
    ofs.open(filename, ostream::binary);
    wav_assert(ofs.is_open(), "couldn't open file for writing: " + filename);
    ofs.write(buf, totsize);
    ofs.close();
    delete[] buf;
}

void wav_data::write_to_buffer(char *buf, uint32_t bufsize)
{
    uint32_t reqsize = nbytes() + HEADER_SIZE;
    wav_assert(bufsize >= reqsize,
               "output buffer is too small " + to_string(bufsize) + " vs " + to_string(reqsize));
    write_header(buf, HEADER_SIZE);
    write_data(buf + HEADER_SIZE, nbytes());
}

void wav_data::write_header(char *buf, uint32_t bufsize)
{
    RiffMainHeader rh;
    rh.dwRiffCC      = nervana::FOURCC('R', 'I', 'F', 'F');
    rh.dwRiffLen     = HEADER_SIZE + nbytes();
    rh.dwWaveID      = nervana::FOURCC('W', 'A', 'V', 'E');

    FmtHeader fh;
    fh.dwFmtCC       = nervana::FOURCC('f', 'm', 't', ' ');
    fh.dwFmtLen      = sizeof(FmtHeader) - 2 * sizeof(uint32_t);
    fh.hwFmtTag      = WAVE_FORMAT_PCM;
    fh.hwChannels    =  data.cols;
    fh.dwSampleRate  = _sample_rate;
    fh.dwBytesPerSec = _sample_rate * data.elemSize();
    fh.hwBlockAlign  = data.elemSize() * data.cols;
    fh.hwBitDepth    = data.elemSize() * 8;

    DataHeader dh;
    dh.dwDataCC      = nervana::FOURCC('d', 'a', 't', 'a');
    dh.dwDataLen     = nbytes();

    // buf.resize();
    assert(bufsize >= HEADER_SIZE);

    memcpy(buf, &rh, sizeof(rh)); buf += sizeof(rh);
    memcpy(buf, &fh, sizeof(fh)); buf += sizeof(fh);
    memcpy(buf, &dh, sizeof(dh));

}

void wav_data::write_data(char* buf, uint32_t bufsize)
{
    assert(bufsize >= nbytes());
    for (int n = 0; n < data.rows; n++) {
        int16_t *ptr = data.ptr<int16_t>(n);
        for (int c = 0; c < data.cols; c++) {
            pack_le(buf, ptr[c], (n * data.cols + c) * sizeof(int16_t));
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
    c->channels = 1;

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
            samples[j] = (int)(sin(t) * 10000);
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

