/*
 Copyright 2016 Intel(R) Nervana(TM)
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "wav_data.hpp"

using namespace std;
using namespace nervana;
using nervana::pack;
using nervana::unpack;

wav_data::wav_data(const char* buf, uint32_t bufsize)
{
    size_t pos = 0;

    wav_assert(bufsize >= HEADER_SIZE, "Header size is too small");

    RiffMainHeader rh;
    FmtHeader      fh;

    memcpy(&rh, buf + pos, sizeof(rh));
    pos += sizeof(rh);
    memcpy(&fh, buf + pos, sizeof(fh));
    pos += sizeof(fh);

    wav_assert(rh.dwRiffCC == nervana::FOURCC('R', 'I', 'F', 'F'), "Unsupported format");
    wav_assert(rh.dwWaveID == nervana::FOURCC('W', 'A', 'V', 'E'), "Unsupported format");
    wav_assert(bufsize >= rh.dwRiffLen, "Buffer not large enough for indicated file size");

    wav_assert(fh.hwFmtTag == WAVE_FORMAT_PCM, "can read only PCM data");
    wav_assert(fh.hwBitDepth == 16, "Ingested waveforms must be 16-bit PCM");
    wav_assert(fh.hwChannels == 1, "Can only handle mono data");
    wav_assert(fh.dwFmtLen >= 16, "PCM format data must be at least 16 bytes");

    // Skip any subchunks between "fmt" and "data".
    while (strncmp(buf + pos, "data", 4) != 0)
    {
        uint32_t chunk_sz = unpack<uint32_t>(buf + pos + 4);
        wav_assert(chunk_sz == 4 || strncmp(buf + pos, "fact", 4), "Malformed fact chunk");
        pos += 4 + sizeof(chunk_sz) + chunk_sz; // chunk tag, chunk size, chunk
    }

    wav_assert(strncmp(buf + pos, "data", 4) == 0, "Expected data tag not found");

    DataHeader dh;
    memcpy(&dh, buf + pos, sizeof(dh));
    pos += sizeof(dh);

    uint32_t num_samples = dh.dwDataLen / fh.hwBlockAlign;
    data.create(num_samples, fh.hwChannels, CV_16SC1);
    _sample_rate = fh.dwSampleRate;

    for (uint32_t n = 0; n < data.rows; ++n)
    {
        for (uint32_t c = 0; c < data.cols; ++c)
        {
            data.at<int16_t>(n, c) = unpack<int16_t>(buf + pos);
            pos += sizeof(int16_t);
        }
    }
}

void wav_data::dump(std::ostream& ostr)
{
    ostr << "sample_rate " << _sample_rate << "\n";
    ostr << "channels x samples " << data.size() << "\n";
    ostr << "bit_depth " << (data.elemSize() * 8) << "\n";
    ostr << "nbytes " << nbytes() << "\n";
}

void wav_data::write_to_file(string filename)
{
    uint32_t totsize = HEADER_SIZE + nbytes();
    char*    buf     = new char[totsize];

    write_to_buffer(buf, totsize);

    std::ofstream ofs;
    ofs.open(filename, ostream::binary);
    wav_assert(ofs.is_open(), "couldn't open file for writing: " + filename);
    ofs.write(buf, totsize);
    ofs.close();
    delete[] buf;
}

void wav_data::write_to_buffer(char* buf, uint32_t bufsize)
{
    uint32_t reqsize = nbytes() + HEADER_SIZE;
    wav_assert(bufsize >= reqsize,
               "output buffer is too small " + to_string(bufsize) + " vs " + to_string(reqsize));
    write_header(buf, HEADER_SIZE);
    write_data(buf + HEADER_SIZE, nbytes());
}

void wav_data::write_header(char* buf, uint32_t bufsize)
{
    RiffMainHeader rh;
    rh.dwRiffCC  = nervana::FOURCC('R', 'I', 'F', 'F');
    rh.dwRiffLen = HEADER_SIZE + nbytes() - 2 * sizeof(uint32_t);
    rh.dwWaveID  = nervana::FOURCC('W', 'A', 'V', 'E');

    FmtHeader fh;
    fh.dwFmtCC       = nervana::FOURCC('f', 'm', 't', ' ');
    fh.dwFmtLen      = sizeof(FmtHeader) - 2 * sizeof(uint32_t);
    fh.hwFmtTag      = WAVE_FORMAT_PCM;
    fh.hwChannels    = data.cols;
    fh.dwSampleRate  = _sample_rate;
    fh.dwBytesPerSec = _sample_rate * data.elemSize();
    fh.hwBlockAlign  = data.elemSize() * data.cols;
    fh.hwBitDepth    = data.elemSize() * 8;

    DataHeader dh;
    dh.dwDataCC  = nervana::FOURCC('d', 'a', 't', 'a');
    dh.dwDataLen = nbytes();

    // buf.resize();
    wav_assert(bufsize >= HEADER_SIZE, "buffer is too small");

    memcpy(buf, &rh, sizeof(rh));
    buf += sizeof(rh);
    memcpy(buf, &fh, sizeof(fh));
    buf += sizeof(fh);
    memcpy(buf, &dh, sizeof(dh));
}

void wav_data::write_data(char* buf, uint32_t bufsize)
{
    wav_assert(bufsize >= nbytes(), "buffer is too small");
    for (int n = 0; n < data.rows; n++)
    {
        int16_t* ptr = data.ptr<int16_t>(n);
        for (int c = 0; c < data.cols; c++)
        {
            pack(buf, ptr[c], (n * data.cols + c) * sizeof(int16_t));
        }
    }
}
