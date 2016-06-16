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

#include "gen_video.hpp"

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

gen_video::gen_video() :
    r{42}
{
}

std::vector<unsigned char> gen_video::render_target( int datumNumber ) {
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
    return rc;
}

std::vector<unsigned char> gen_video::render_datum( int datumNumber ) {
    return encode(2000);    // two second clip
}

vector<unsigned char> gen_video::encode(int duration) {
    AVCodec *codec;
    AVCodecContext *c= NULL;
    int i, out_size, size, x, y, outbuf_size;
    AVFrame *picture;
    uint8_t *outbuf, *picture_buf;
    vector<unsigned char> rc;

    /* find the mpeg1 video encoder */
    codec = avcodec_find_encoder(CODEC_ID_MPEG1VIDEO);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c= avcodec_alloc_context3(codec);
    picture= avcodec_alloc_frame();

    /* put sample parameters */
    c->bit_rate = 400000;
    /* resolution must be a multiple of two */
    c->width = 352;
    c->height = 288;
    /* frames per second */
    c->time_base= (AVRational){1,25};
    c->gop_size = 10; /* emit one intra frame every ten frames */
    c->max_b_frames=1;
    c->pix_fmt = PIX_FMT_YUV420P;

    /* open it */
    if (avcodec_open2(c, codec, nullptr) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    /* alloc image and output buffer */
    outbuf_size = 100000;
    outbuf = (uint8_t*)malloc(outbuf_size);
    size = c->width * c->height;
    picture_buf = (uint8_t*)malloc((size * 3) / 2); /* size for YUV 420 */

    picture->data[0] = picture_buf;
    picture->data[1] = picture->data[0] + size;
    picture->data[2] = picture->data[1] + size / 4;
    picture->linesize[0] = c->width;
    picture->linesize[1] = c->width / 2;
    picture->linesize[2] = c->width / 2;

    /* encode 1 second of video */
    int frames = ((float)duration / 1000.) / ((float)(c->time_base.num) / (float)(c->time_base.den));
    for(i=0;i<frames;i++) {
        /* prepare a dummy image */
        /* Y */
        for(y=0;y<c->height;y++) {
            for(x=0;x<c->width;x++) {
                picture->data[0][y * picture->linesize[0] + x] = x + y + i * 3;
            }
        }

        /* Cb and Cr */
        for(y=0;y<c->height/2;y++) {
            for(x=0;x<c->width/2;x++) {
                picture->data[1][y * picture->linesize[1] + x] = 128 + y + i * 2;
                picture->data[2][y * picture->linesize[2] + x] = 64 + x + i * 5;
            }
        }

        /* encode the image */
        out_size = avcodec_encode_video(c, outbuf, outbuf_size, picture);
        for(int i=0; i<out_size; i++) { rc.push_back(outbuf[i]); }
    }

    /* get the delayed frames */
    for(; out_size; i++) {
        out_size = avcodec_encode_video(c, outbuf, outbuf_size, NULL);
        for(int i=0; i<out_size; i++) { rc.push_back(outbuf[i]); }
    }

    /* add sequence end code to have a real mpeg file */
    outbuf[0] = 0x00;
    outbuf[1] = 0x00;
    outbuf[2] = 0x01;
    outbuf[3] = 0xb7;
    for(int i=0; i<out_size; i++) { rc.push_back(outbuf[i]); }
    free(picture_buf);
    free(outbuf);

    avcodec_close(c);
    av_free(c);
    av_free(picture);

    return rc;
}

void gen_video::encode(const std::string& filename, int duration)
{
    FILE* f;
    f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename.c_str());
        exit(1);
    }

    vector<unsigned char> data = encode(duration);
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
}

/*
 * Video decoding example
 */

void gen_video::pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename)
{
    FILE *f;
    int i;

    f=fopen(filename,"w");
    fprintf(f,"P5\n%d %d\n%d\n",xsize,ysize,255);
    for(i=0;i<ysize;i++)
        fwrite(buf + i * wrap,1,xsize,f);
    fclose(f);
}

void gen_video::decode(const std::string& outfilename, const std::string& filename)
{
    AVCodec *codec;
    AVCodecContext *c= NULL;
    int frame, got_picture, len;
    FILE *f;
    AVFrame *picture;
    uint8_t inbuf[INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];
    char buf[1024];
    AVPacket avpkt;

    av_init_packet(&avpkt);

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged mpeg streams) */
    memset(inbuf + INBUF_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    printf("Video decoding\n");

    /* find the mpeg1 video decoder */
    codec = avcodec_find_decoder(CODEC_ID_MPEG1VIDEO);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c= avcodec_alloc_context3(codec);
    picture= avcodec_alloc_frame();

    if(codec->capabilities&CODEC_CAP_TRUNCATED)
        c->flags|= CODEC_FLAG_TRUNCATED; /* we do not send complete frames */

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */

    /* open it */
    if (avcodec_open2(c, codec, nullptr) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    /* the codec gives us the frame size, in samples */

    f = fopen(filename.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename.c_str());
        exit(1);
    }

    frame = 0;
    for(;;) {
        avpkt.size = fread(inbuf, 1, INBUF_SIZE, f);
        if (avpkt.size == 0)
            break;

        /* NOTE1: some codecs are stream based (mpegvideo, mpegaudio)
           and this is the only method to use them because you cannot
           know the compressed data size before analysing it.

           BUT some other codecs (msmpeg4, mpeg4) are inherently frame
           based, so you must call them with all the data for one
           frame exactly. You must also initialize 'width' and
           'height' before initializing them. */

        /* NOTE2: some codecs allow the raw parameters (frame size,
           sample rate) to be changed at any frame. We handle this, so
           you should also take care of it */

        /* here, we use a stream based decoder (mpeg1video), so we
           feed decoder and see if it could decode a frame */
        avpkt.data = inbuf;
        while (avpkt.size > 0) {
            len = avcodec_decode_video2(c, picture, &got_picture, &avpkt);
            if (len < 0) {
                fprintf(stderr, "Error while decoding frame %d\n", frame);
                exit(1);
            }
            if (got_picture) {
                printf("saving frame %3d\n", frame);
                fflush(stdout);

                /* the picture is allocated by the decoder. no need to
                   free it */
                snprintf(buf, sizeof(buf), outfilename.c_str(), frame);
                pgm_save(picture->data[0], picture->linesize[0],
                         c->width, c->height, buf);
                frame++;
            }
            avpkt.size -= len;
            avpkt.data += len;
        }
    }

    /* some codecs, such as MPEG, transmit the I and P frame with a
       latency of one frame. You must do the following to have a
       chance to get the last frame of the video */
    avpkt.data = NULL;
    avpkt.size = 0;
    len = avcodec_decode_video2(c, picture, &got_picture, &avpkt);
    if (got_picture) {
        printf("saving last frame %3d\n", frame);
        fflush(stdout);

        /* the picture is allocated by the decoder. no need to
           free it */
        snprintf(buf, sizeof(buf), outfilename.c_str(), frame);
        pgm_save(picture->data[0], picture->linesize[0],
                 c->width, c->height, buf);
        frame++;
    }

    fclose(f);

    avcodec_close(c);
    av_free(c);
    av_free(picture);
    printf("\n");
}
