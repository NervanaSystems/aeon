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

gen_audio::gen_audio() :
    r{42}
{
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

/*
 * Audio encoding example
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

//int main(int argc, char **argv)
//{
//    const char *filename;

//    /* must be called before using avcodec lib */
////    avcodec_init();

//    /* register all the codecs */
//    avcodec_register_all();

//    if (argc <= 1) {
//        audio_encode_example("/tmp/test.mp2");
//        audio_decode_example("/tmp/test.sw", "/tmp/test.mp2");

//        video_encode_example("/tmp/test.mpg");
//        filename = "/tmp/test.mpg";
//    } else {
//        filename = argv[1];
//    }

//    //    audio_decode_example("/tmp/test.sw", filename);
//    video_decode_example("/tmp/test%d.pgm", filename);

//    return 0;
//}
