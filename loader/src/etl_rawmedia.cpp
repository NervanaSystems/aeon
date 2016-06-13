#include "etl_rawmedia.hpp"

using namespace std;
using namespace nervana;

void rawmedia::decoded::copy_data(char* buf, int buf_size)
{
    if (_data_size * (int) _bufs.size() > buf_size) {
        stringstream ss;
        ss << "Buffer too small to copy decoded data. Buffer size " <<
               buf_size << " Data size " << _data_size * _bufs.size();
        throw runtime_error(ss.str());
    }

    for (uint i = 0; i < _bufs.size(); i++) {
        memcpy(buf, _bufs[i], _data_size);
        buf += _data_size;
    }
}

void rawmedia::decoded::add_bufs(int count, int size)
{
    for (int i = 0; i < count; i++) {
        _bufs.push_back(new char[size]);
    }
    _buf_size = size;
}

void rawmedia::decoded::fill_bufs(char** frames, int frame_size)
{
    for (uint i = 0; i < _bufs.size(); i++) {
        memcpy(_bufs[i] + _data_size, frames[i], frame_size);
    }
    _data_size += frame_size;
}

void rawmedia::decoded::grow_bufs(int grow)
{
    for (uint i = 0; i < _bufs.size(); i++) {
        char* buf = new char[_buf_size + grow];
        memcpy(buf, _bufs[i], _data_size);
        delete[] _bufs[i];
        _bufs[i] = buf;
    }
    _buf_size += grow;
}

shared_ptr<rawmedia::decoded> rawmedia::extractor::extract(const char* item, int item_size)
{
    auto _raw = make_shared<rawmedia::decoded>();

    _format = avformat_alloc_context();
    if (_format == 0) {
        throw runtime_error("Could not get context for decoding");
    }
    uchar* item_copy = (uchar*) av_malloc(item_size);
    if (item_copy == 0) {
        throw runtime_error("Could not allocate memory");
    }

    memcpy(item_copy, item, item_size);
    _format->pb = avio_alloc_context(item_copy, item_size, 0, 0, 0, 0, 0);

    if (avformat_open_input(&_format , "", 0, 0) < 0) {
        throw runtime_error("Could not open input for decoding");
    }

    if (avformat_find_stream_info(_format, 0) < 0) {
        throw runtime_error("Could not find media information");
    }

    _codec = _format->streams[0]->codec;
    int stream = av_find_best_stream(_format, _avmedia_type, -1, -1, 0, 0);

    if (stream < 0) {
        throw runtime_error("Could not find media stream in input");
    }

    if (avcodec_open2(_codec, avcodec_find_decoder(_codec->codec_id), 0) < 0) {
        throw runtime_error("Could not open decoder");
    }

    if (_raw->size() == 0) {
        _raw->add_bufs(_codec->channels, item_size);
    } else {
        _raw->reset();
    }

    _raw->set_sample_size(av_get_bytes_per_sample(_codec->sample_fmt));

    assert(_raw->get_sample_size() >= 0);

    AVPacket packet;
    while (av_read_frame(_format, &packet) >= 0) {
        decode_frame(&packet, _raw, stream, item_size);
    }

    avcodec_close(_codec);
    av_free(_format->pb->buffer);
    av_free(_format->pb);
    avformat_close_input(&_format);
    return _raw;
}

void rawmedia::extractor::decode_frame(AVPacket* packet,
                                       shared_ptr<rawmedia::decoded> raw,
                                       int stream,
                                       int item_size)
{
    int frame_done;
    if (packet->stream_index == stream) {
        AVFrame* frame = av_frame_alloc();
        int result = 0;
        if (_mediaType == AVMEDIA_TYPE_AUDIO) {
            result = avcodec_decode_audio4(_codec, frame, &frame_done, packet);
        } else {
            throw runtime_error("Unsupported media");
        }

        if (result < 0) {
            throw runtime_error("Could not decode media stream");
        }

        if (frame_done == true) {
            int frame_size = frame->nb_samples * raw->get_sample_size();
            if (raw->get_buf_size() < raw->get_data_size() + frame_size) {
                raw->grow_bufs(item_size);
            }
            raw->fill_bufs((char**) frame->data, frame_size);
        }
        av_frame_free(&frame);
    }
    av_free_packet(packet);
}
