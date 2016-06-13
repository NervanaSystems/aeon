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

#ifdef HAVE_AV_CONFIG_H
#undef HAVE_AV_CONFIG_H
#endif

#include "libavcodec/avcodec.h"
#include "libavutil/mathematics.h"

#define INBUF_SIZE 4096
#define AUDIO_INBUF_SIZE 20480
#define AUDIO_REFILL_THRESH 4096

/*
 * Audio encoding example
 */
static void audio_encode_example(const char *filename)
{
    AVCodec *codec;
    AVCodecContext *c= NULL;
    int frame_size, i, j, out_size, outbuf_size;
    FILE *f;
    short *samples;
    float t, tincr;
    uint8_t *outbuf;

    printf("Audio encoding\n");

    /* find the MP2 encoder */
    codec = avcodec_find_encoder(CODEC_ID_MP2);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c= avcodec_alloc_context();

    /* put sample parameters */
    c->bit_rate = 64000;
    c->sample_rate = 44100;
    c->channels = 2;

    /* open it */
    if (avcodec_open(c, codec) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    /* the codec gives us the frame size, in samples */
    frame_size = c->frame_size;
    samples = malloc(frame_size * 2 * c->channels);
    outbuf_size = 10000;
    outbuf = malloc(outbuf_size);

    f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename);
        exit(1);
    }

    /* encode a single tone sound */
    t = 0;
    tincr = 2 * M_PI * 440.0 / c->sample_rate;
    for(i=0;i<200;i++) {
        for(j=0;j<frame_size;j++) {
            samples[2*j] = (int)(sin(t) * 10000);
            samples[2*j+1] = samples[2*j];
            t += tincr;
        }
        /* encode the samples */
        out_size = avcodec_encode_audio(c, outbuf, outbuf_size, samples);
        fwrite(outbuf, 1, out_size, f);
    }
    fclose(f);
    free(outbuf);
    free(samples);

    avcodec_close(c);
    av_free(c);
}

/*
 * Audio decoding.
 */
static void audio_decode_example(const char *outfilename, const char *filename)
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

    c= avcodec_alloc_context();

    /* open it */
    if (avcodec_open(c, codec) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    outbuf = malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE);

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename);
        exit(1);
    }
    outfile = fopen(outfilename, "wb");
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

/*
 * Video encoding example
 */
static void video_encode_example(const char *filename)
{
    AVCodec *codec;
    AVCodecContext *c= NULL;
    int i, out_size, size, x, y, outbuf_size;
    FILE *f;
    AVFrame *picture;
    uint8_t *outbuf, *picture_buf;

    printf("Video encoding\n");

    /* find the mpeg1 video encoder */
    codec = avcodec_find_encoder(CODEC_ID_MPEG1VIDEO);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    c= avcodec_alloc_context();
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
    if (avcodec_open(c, codec) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename);
        exit(1);
    }

    /* alloc image and output buffer */
    outbuf_size = 100000;
    outbuf = malloc(outbuf_size);
    size = c->width * c->height;
    picture_buf = malloc((size * 3) / 2); /* size for YUV 420 */

    picture->data[0] = picture_buf;
    picture->data[1] = picture->data[0] + size;
    picture->data[2] = picture->data[1] + size / 4;
    picture->linesize[0] = c->width;
    picture->linesize[1] = c->width / 2;
    picture->linesize[2] = c->width / 2;

    /* encode 1 second of video */
    for(i=0;i<25;i++) {
        fflush(stdout);
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
        printf("encoding frame %3d (size=%5d)\n", i, out_size);
        fwrite(outbuf, 1, out_size, f);
    }

    /* get the delayed frames */
    for(; out_size; i++) {
        fflush(stdout);

        out_size = avcodec_encode_video(c, outbuf, outbuf_size, NULL);
        printf("write frame %3d (size=%5d)\n", i, out_size);
        fwrite(outbuf, 1, out_size, f);
    }

    /* add sequence end code to have a real mpeg file */
    outbuf[0] = 0x00;
    outbuf[1] = 0x00;
    outbuf[2] = 0x01;
    outbuf[3] = 0xb7;
    fwrite(outbuf, 1, 4, f);
    fclose(f);
    free(picture_buf);
    free(outbuf);

    avcodec_close(c);
    av_free(c);
    av_free(picture);
    printf("\n");
}

/*
 * Video decoding example
 */

static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
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

static void video_decode_example(const char *outfilename, const char *filename)
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

    c= avcodec_alloc_context();
    picture= avcodec_alloc_frame();

    if(codec->capabilities&CODEC_CAP_TRUNCATED)
        c->flags|= CODEC_FLAG_TRUNCATED; /* we do not send complete frames */

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */

    /* open it */
    if (avcodec_open(c, codec) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    /* the codec gives us the frame size, in samples */

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename);
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
                snprintf(buf, sizeof(buf), outfilename, frame);
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
        snprintf(buf, sizeof(buf), outfilename, frame);
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

int main(int argc, char **argv)
{
    const char *filename;

    /* must be called before using avcodec lib */
    avcodec_init();

    /* register all the codecs */
    avcodec_register_all();

    if (argc <= 1) {
        audio_encode_example("/tmp/test.mp2");
        audio_decode_example("/tmp/test.sw", "/tmp/test.mp2");

        video_encode_example("/tmp/test.mpg");
        filename = "/tmp/test.mpg";
    } else {
        filename = argv[1];
    }

    //    audio_decode_example("/tmp/test.sw", filename);
    video_decode_example("/tmp/test%d.pgm", filename);

    return 0;
}



































///*
// * Libavformat API example: Output a media file in any supported
// * libavformat format. The default codecs are used.
// *
// * Copyright (c) 2003 Fabrice Bellard
// *
// * Permission is hereby granted, free of charge, to any person obtaining a copy
// * of this software and associated documentation files (the "Software"), to deal
// * in the Software without restriction, including without limitation the rights
// * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// * copies of the Software, and to permit persons to whom the Software is
// * furnished to do so, subject to the following conditions:
// *
// * The above copyright notice and this permission notice shall be included in
// * all copies or substantial portions of the Software.
// *
// * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// * THE SOFTWARE.
// */
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <math.h>

//extern "C" {
//    #include "libavformat/avformat.h"
//    #include "libswscale/swscale.h"
//}

//#undef exit

///* 5 seconds stream duration */
//#define STREAM_DURATION   5.0
//#define STREAM_FRAME_RATE 25 /* 25 images/s */
//#define STREAM_NB_FRAMES  ((int)(STREAM_DURATION * STREAM_FRAME_RATE))
//#define STREAM_PIX_FMT PIX_FMT_YUV420P /* default pix_fmt */

//static int sws_flags = SWS_BICUBIC;

///**************************************************************/
///* audio output */

//float t, tincr, tincr2;
//int16_t *samples;
//uint8_t *audio_outbuf;
//int audio_outbuf_size;
//int audio_input_frame_size;

///*
// * add an audio output stream
// */
//static AVStream *add_audio_stream(AVFormatContext *oc, enum CodecID codec_id)
//{
//    AVCodecContext *c;
//    AVStream *st;

//    st = av_new_stream(oc, 1);
//    if (!st) {
//        fprintf(stderr, "Could not alloc stream\n");
//        exit(1);
//    }

//    c = st->codec;
//    c->codec_id = codec_id;
//    c->codec_type = AVMEDIA_TYPE_AUDIO;

//    /* put sample parameters */
//    c->bit_rate = 64000;
//    c->sample_rate = 44100;
//    c->channels = 2;

//    // some formats want stream headers to be separate
//    if(oc->oformat->flags & AVFMT_GLOBALHEADER)
//        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

//    return st;
//}

//static void open_audio(AVFormatContext *oc, AVStream *st)
//{
//    AVCodecContext *c;
//    AVCodec *codec;

//    c = st->codec;

//    /* find the audio encoder */
//    codec = avcodec_find_encoder(c->codec_id);
//    if (!codec) {
//        fprintf(stderr, "codec not found\n");
//        exit(1);
//    }

//    /* open it */
//    if (avcodec_open(c, codec) < 0) {
//        fprintf(stderr, "could not open codec\n");
//        exit(1);
//    }

//    /* init signal generator */
//    t = 0;
//    tincr = 2 * M_PI * 110.0 / c->sample_rate;
//    /* increment frequency by 110 Hz per second */
//    tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;

//    audio_outbuf_size = 10000;
//    audio_outbuf = av_malloc(audio_outbuf_size);

//    /* ugly hack for PCM codecs (will be removed ASAP with new PCM
//       support to compute the input frame size in samples */
//    if (c->frame_size <= 1) {
//        audio_input_frame_size = audio_outbuf_size / c->channels;
//        switch(st->codec->codec_id) {
//        case CODEC_ID_PCM_S16LE:
//        case CODEC_ID_PCM_S16BE:
//        case CODEC_ID_PCM_U16LE:
//        case CODEC_ID_PCM_U16BE:
//            audio_input_frame_size >>= 1;
//            break;
//        default:
//            break;
//        }
//    } else {
//        audio_input_frame_size = c->frame_size;
//    }
//    samples = av_malloc(audio_input_frame_size * 2 * c->channels);
//}

///* prepare a 16 bit dummy audio frame of 'frame_size' samples and
//   'nb_channels' channels */
//static void get_audio_frame(int16_t *samples, int frame_size, int nb_channels)
//{
//    int j, i, v;
//    int16_t *q;

//    q = samples;
//    for(j=0;j<frame_size;j++) {
//        v = (int)(sin(t) * 10000);
//        for(i = 0; i < nb_channels; i++)
//            *q++ = v;
//        t += tincr;
//        tincr += tincr2;
//    }
//}

//static void write_audio_frame(AVFormatContext *oc, AVStream *st)
//{
//    AVCodecContext *c;
//    AVPacket pkt;
//    av_init_packet(&pkt);

//    c = st->codec;

//    get_audio_frame(samples, audio_input_frame_size, c->channels);

//    pkt.size= avcodec_encode_audio(c, audio_outbuf, audio_outbuf_size, samples);

//    if (c->coded_frame && c->coded_frame->pts != AV_NOPTS_VALUE)
//        pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, st->time_base);
//    pkt.flags |= AV_PKT_FLAG_KEY;
//    pkt.stream_index= st->index;
//    pkt.data= audio_outbuf;

//    /* write the compressed frame in the media file */
//    if (av_interleaved_write_frame(oc, &pkt) != 0) {
//        fprintf(stderr, "Error while writing audio frame\n");
//        exit(1);
//    }
//}

//static void close_audio(AVFormatContext *oc, AVStream *st)
//{
//    avcodec_close(st->codec);

//    av_free(samples);
//    av_free(audio_outbuf);
//}

///**************************************************************/
///* video output */

//AVFrame *picture, *tmp_picture;
//uint8_t *video_outbuf;
//int frame_count, video_outbuf_size;

///* add a video output stream */
//static AVStream *add_video_stream(AVFormatContext *oc, enum CodecID codec_id)
//{
//    AVCodecContext *c;
//    AVStream *st;

//    st = av_new_stream(oc, 0);
//    if (!st) {
//        fprintf(stderr, "Could not alloc stream\n");
//        exit(1);
//    }

//    c = st->codec;
//    c->codec_id = codec_id;
//    c->codec_type = AVMEDIA_TYPE_VIDEO;

//    /* put sample parameters */
//    c->bit_rate = 400000;
//    /* resolution must be a multiple of two */
//    c->width = 352;
//    c->height = 288;
//    /* time base: this is the fundamental unit of time (in seconds) in terms
//       of which frame timestamps are represented. for fixed-fps content,
//       timebase should be 1/framerate and timestamp increments should be
//       identically 1. */
//    c->time_base.den = STREAM_FRAME_RATE;
//    c->time_base.num = 1;
//    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
//    c->pix_fmt = STREAM_PIX_FMT;
//    if (c->codec_id == CODEC_ID_MPEG2VIDEO) {
//        /* just for testing, we also add B frames */
//        c->max_b_frames = 2;
//    }
//    if (c->codec_id == CODEC_ID_MPEG1VIDEO){
//        /* Needed to avoid using macroblocks in which some coeffs overflow.
//           This does not happen with normal video, it just happens here as
//           the motion of the chroma plane does not match the luma plane. */
//        c->mb_decision=2;
//    }
//    // some formats want stream headers to be separate
//    if(oc->oformat->flags & AVFMT_GLOBALHEADER)
//        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

//    return st;
//}

//static AVFrame *alloc_picture(enum PixelFormat pix_fmt, int width, int height)
//{
//    AVFrame *picture;
//    uint8_t *picture_buf;
//    int size;

//    picture = avcodec_alloc_frame();
//    if (!picture)
//        return NULL;
//    size = avpicture_get_size(pix_fmt, width, height);
//    picture_buf = av_malloc(size);
//    if (!picture_buf) {
//        av_free(picture);
//        return NULL;
//    }
//    avpicture_fill((AVPicture *)picture, picture_buf,
//                   pix_fmt, width, height);
//    return picture;
//}

//static void open_video(AVFormatContext *oc, AVStream *st)
//{
//    AVCodec *codec;
//    AVCodecContext *c;

//    c = st->codec;

//    /* find the video encoder */
//    codec = avcodec_find_encoder(c->codec_id);
//    if (!codec) {
//        fprintf(stderr, "codec not found\n");
//        exit(1);
//    }

//    /* open the codec */
//    if (avcodec_open(c, codec) < 0) {
//        fprintf(stderr, "could not open codec\n");
//        exit(1);
//    }

//    video_outbuf = NULL;
//    if (!(oc->oformat->flags & AVFMT_RAWPICTURE)) {
//        /* allocate output buffer */
//        /* XXX: API change will be done */
//        /* buffers passed into lav* can be allocated any way you prefer,
//           as long as they're aligned enough for the architecture, and
//           they're freed appropriately (such as using av_free for buffers
//           allocated with av_malloc) */
//        video_outbuf_size = 200000;
//        video_outbuf = av_malloc(video_outbuf_size);
//    }

//    /* allocate the encoded raw picture */
//    picture = alloc_picture(c->pix_fmt, c->width, c->height);
//    if (!picture) {
//        fprintf(stderr, "Could not allocate picture\n");
//        exit(1);
//    }

//    /* if the output format is not YUV420P, then a temporary YUV420P
//       picture is needed too. It is then converted to the required
//       output format */
//    tmp_picture = NULL;
//    if (c->pix_fmt != PIX_FMT_YUV420P) {
//        tmp_picture = alloc_picture(PIX_FMT_YUV420P, c->width, c->height);
//        if (!tmp_picture) {
//            fprintf(stderr, "Could not allocate temporary picture\n");
//            exit(1);
//        }
//    }
//}

///* prepare a dummy image */
//static void fill_yuv_image(AVFrame *pict, int frame_index, int width, int height)
//{
//    int x, y, i;

//    i = frame_index;

//    /* Y */
//    for(y=0;y<height;y++) {
//        for(x=0;x<width;x++) {
//            pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;
//        }
//    }

//    /* Cb and Cr */
//    for(y=0;y<height/2;y++) {
//        for(x=0;x<width/2;x++) {
//            pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
//            pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
//        }
//    }
//}

//static void write_video_frame(AVFormatContext *oc, AVStream *st)
//{
//    int out_size, ret;
//    AVCodecContext *c;
//    static struct SwsContext *img_convert_ctx;

//    c = st->codec;

//    if (frame_count >= STREAM_NB_FRAMES) {
//        /* no more frame to compress. The codec has a latency of a few
//           frames if using B frames, so we get the last frames by
//           passing the same picture again */
//    } else {
//        if (c->pix_fmt != PIX_FMT_YUV420P) {
//            /* as we only generate a YUV420P picture, we must convert it
//               to the codec pixel format if needed */
//            if (img_convert_ctx == NULL) {
//                img_convert_ctx = sws_getContext(c->width, c->height,
//                                                 PIX_FMT_YUV420P,
//                                                 c->width, c->height,
//                                                 c->pix_fmt,
//                                                 sws_flags, NULL, NULL, NULL);
//                if (img_convert_ctx == NULL) {
//                    fprintf(stderr, "Cannot initialize the conversion context\n");
//                    exit(1);
//                }
//            }
//            fill_yuv_image(tmp_picture, frame_count, c->width, c->height);
//            sws_scale(img_convert_ctx, tmp_picture->data, tmp_picture->linesize,
//                      0, c->height, picture->data, picture->linesize);
//        } else {
//            fill_yuv_image(picture, frame_count, c->width, c->height);
//        }
//    }


//    if (oc->oformat->flags & AVFMT_RAWPICTURE) {
//        /* raw video case. The API will change slightly in the near
//           futur for that */
//        AVPacket pkt;
//        av_init_packet(&pkt);

//        pkt.flags |= AV_PKT_FLAG_KEY;
//        pkt.stream_index= st->index;
//        pkt.data= (uint8_t *)picture;
//        pkt.size= sizeof(AVPicture);

//        ret = av_interleaved_write_frame(oc, &pkt);
//    } else {
//        /* encode the image */
//        out_size = avcodec_encode_video(c, video_outbuf, video_outbuf_size, picture);
//        /* if zero size, it means the image was buffered */
//        if (out_size > 0) {
//            AVPacket pkt;
//            av_init_packet(&pkt);

//            if (c->coded_frame->pts != AV_NOPTS_VALUE)
//                pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, st->time_base);
//            if(c->coded_frame->key_frame)
//                pkt.flags |= AV_PKT_FLAG_KEY;
//            pkt.stream_index= st->index;
//            pkt.data= video_outbuf;
//            pkt.size= out_size;

//            /* write the compressed frame in the media file */
//            ret = av_interleaved_write_frame(oc, &pkt);
//        } else {
//            ret = 0;
//        }
//    }
//    if (ret != 0) {
//        fprintf(stderr, "Error while writing video frame\n");
//        exit(1);
//    }
//    frame_count++;
//}

//static void close_video(AVFormatContext *oc, AVStream *st)
//{
//    avcodec_close(st->codec);
//    av_free(picture->data[0]);
//    av_free(picture);
//    if (tmp_picture) {
//        av_free(tmp_picture->data[0]);
//        av_free(tmp_picture);
//    }
//    av_free(video_outbuf);
//}

///**************************************************************/
///* media file output */

//int main(int argc, char **argv)
//{
//    const char *filename;
//    AVOutputFormat *fmt;
//    AVFormatContext *oc;
//    AVStream *audio_st, *video_st;
//    double audio_pts, video_pts;
//    int i;

//    /* initialize libavcodec, and register all codecs and formats */
//    av_register_all();

//    if (argc != 2) {
//        printf("usage: %s output_file\n"
//               "API example program to output a media file with libavformat.\n"
//               "The output format is automatically guessed according to the file extension.\n"
//               "Raw images can also be output by using '%%d' in the filename\n"
//               "\n", argv[0]);
//        exit(1);
//    }

//    filename = argv[1];

//    /* auto detect the output format from the name. default is
//       mpeg. */
//    fmt = av_guess_format(NULL, filename, NULL);
//    if (!fmt) {
//        printf("Could not deduce output format from file extension: using MPEG.\n");
//        fmt = av_guess_format("mpeg", NULL, NULL);
//    }
//    if (!fmt) {
//        fprintf(stderr, "Could not find suitable output format\n");
//        exit(1);
//    }

//    /* allocate the output media context */
//    oc = avformat_alloc_context();
//    if (!oc) {
//        fprintf(stderr, "Memory error\n");
//        exit(1);
//    }
//    oc->oformat = fmt;
//    snprintf(oc->filename, sizeof(oc->filename), "%s", filename);

//    /* add the audio and video streams using the default format codecs
//       and initialize the codecs */
//    video_st = NULL;
//    audio_st = NULL;
//    if (fmt->video_codec != CODEC_ID_NONE) {
//        video_st = add_video_stream(oc, fmt->video_codec);
//    }
//    if (fmt->audio_codec != CODEC_ID_NONE) {
//        audio_st = add_audio_stream(oc, fmt->audio_codec);
//    }

//    /* set the output parameters (must be done even if no
//       parameters). */
//    if (av_set_parameters(oc, NULL) < 0) {
//        fprintf(stderr, "Invalid output format parameters\n");
//        exit(1);
//    }

//    dump_format(oc, 0, filename, 1);

//    /* now that all the parameters are set, we can open the audio and
//       video codecs and allocate the necessary encode buffers */
//    if (video_st)
//        open_video(oc, video_st);
//    if (audio_st)
//        open_audio(oc, audio_st);

//    /* open the output file, if needed */
//    if (!(fmt->flags & AVFMT_NOFILE)) {
//        if (url_fopen(&oc->pb, filename, URL_WRONLY) < 0) {
//            fprintf(stderr, "Could not open '%s'\n", filename);
//            exit(1);
//        }
//    }

//    /* write the stream header, if any */
//    av_write_header(oc);

//    for(;;) {
//        /* compute current audio and video time */
//        if (audio_st)
//            audio_pts = (double)audio_st->pts.val * audio_st->time_base.num / audio_st->time_base.den;
//        else
//            audio_pts = 0.0;

//        if (video_st)
//            video_pts = (double)video_st->pts.val * video_st->time_base.num / video_st->time_base.den;
//        else
//            video_pts = 0.0;

//        if ((!audio_st || audio_pts >= STREAM_DURATION) &&
//            (!video_st || video_pts >= STREAM_DURATION))
//            break;

//        /* write interleaved audio and video frames */
//        if (!video_st || (video_st && audio_st && audio_pts < video_pts)) {
//            write_audio_frame(oc, audio_st);
//        } else {
//            write_video_frame(oc, video_st);
//        }
//    }

//    /* write the trailer, if any.  the trailer must be written
//     * before you close the CodecContexts open when you wrote the
//     * header; otherwise write_trailer may try to use memory that
//     * was freed on av_codec_close() */
//    av_write_trailer(oc);

//    /* close each codec */
//    if (video_st)
//        close_video(oc, video_st);
//    if (audio_st)
//        close_audio(oc, audio_st);

//    /* free the streams */
//    for(i = 0; i < oc->nb_streams; i++) {
//        av_freep(&oc->streams[i]->codec);
//        av_freep(&oc->streams[i]);
//    }

//    if (!(fmt->flags & AVFMT_NOFILE)) {
//        /* close the output file */
//        url_fclose(oc->pb);
//    }

//    /* free the stream */
//    av_free(oc);

//    return 0;
//}
