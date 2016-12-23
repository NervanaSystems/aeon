/*
 Copyright 2016 Nervana Systems Inc.
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

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "bstream.hpp"

namespace nervana
{
    namespace tiff
    {
        class file_header;
        class directory_entry;
        class reader;

        enum class data_type
        {
            BYTE = 1,
            ASCII = 2,
            SHORT = 3,
            LONG = 4,
            RATIONAL = 5,
            SBYTE = 6,
            UNDEFINED = 7,
            SSHORT = 8,
            SLONG = 9,
            SRATIONAL = 10,
            FLOAT = 11,
            DOUBLE = 12
        };

        enum class tag_type : uint16_t
        {
            NewSubfileType                = 254,
            SubfileType                   = 255,
            ImageWidth                    = 256,
            ImageLength                   = 257,
            BitsPerSample                 = 258,
            Compression                   = 259,
            PhotometricInterpretation     = 262,
            Threshholding                 = 263,
            CellWidth                     = 264,
            CellLength                    = 265,
            FillOrder                     = 266,
            DocumentName                  = 269,
            ImageDescription              = 270,
            Make                          = 271,
            Model                         = 272,
            StripOffsets                  = 273,
            Orientation                   = 274,
            SamplesPerPixel               = 277,
            RowsPerStrip                  = 278,
            StripByteCounts               = 279,
            MinSampleValue                = 280,
            MaxSampleValue                = 281,
            XResolution                   = 282,
            YResolution                   = 283,
            PlanarConfiguration           = 284,
            PageName                      = 285,
            XPosition                     = 286,
            YPosition                     = 287,
            FreeOffsets                   = 288,
            FreeByteCounts                = 289,
            GrayResponseUnit              = 290,
            GrayResponseCurve             = 291,
            T4Options                     = 292,
            T6Options                     = 293,
            ResolutionUnit                = 296,
            PageNumber                    = 297,
            TransferFunction              = 301,
            Software                      = 305,
            DateTime                      = 306,
            Artist                        = 315,
            HostComputer                  = 316,
            Predictor                     = 317,
            WhitePoint                    = 318,
            PrimaryChromaticities         = 319,
            ColorMap                      = 320,
            HalftoneHints                 = 321,
            TileWidth                     = 322,
            TileLength                    = 323,
            TileOffsets                   = 324,
            TileByteCounts                = 325,
            InkSet                        = 332,
            InkNames                      = 333,
            NumberOfInks                  = 334,
            DotRange                      = 336,
            TargetPrinter                 = 337,
            ExtraSamples                  = 338,
            SampleFormat                  = 339,
            SMinSampleValue               = 340,
            SMaxSampleValue               = 341,
            TransferRange                 = 342,
            JPEGProc                      = 512,
            JPEGInterchangeFormat         = 513,
            JPEGInterchangeFormatLngth    = 514,
            JPEGRestartInterval           = 515,
            JPEGLosslessPredictors        = 517,
            JPEGPointTransforms           = 518,
            JPEGQTables                   = 519,
            JPEGDCTables                  = 520,
            JPEGACTables                  = 521,
            YCbCrCoefficients             = 529,
            YCbCrSubSampling              = 530,
            YCbCrPositioning              = 531,
            ReferenceBlackWhite           = 532,
            Copyright                     = 33432
        };

        enum class compression_t
        {
            Uncompressed = 1,
            CCITT_1D     = 2,
            Group_3_FAX  = 3,
            Group_4_FAX  = 4,
            LZW          = 5,
            JPEG         = 6,
            PackBits     = 32773
        };

        enum class photometric_t
        {
            WhiteIsZero         = 0,
            BlackIsZero         = 1,
            RGB                 = 2,
            RGB_Palette         = 3,
            Transparency_mask   = 4,
            CMYK                = 5,
            YCbCr               = 6,
            CIELab              = 8
        };

        bool is_tiff(const char* data, size_t size);
        std::ostream& operator<<(std::ostream&, tag_type);
        std::ostream& operator<<(std::ostream& out, compression_t v);
        std::ostream& operator<<(std::ostream& out, photometric_t v);
    }
}

class nervana::tiff::file_header
{
public:
    file_header(bstream_base& bs);

    uint16_t    byte_order;
    uint16_t    file_id;
    uint32_t    ifd_offset;
};

class nervana::tiff::directory_entry
{
public:
    static directory_entry read(bstream_base& bs);
    cv::Mat read_image(bstream_base& bs) const;

    size_t              image_width;
    size_t              image_length;
    compression_t       compression;
    photometric_t       photometric;
    size_t              strip_offsets_count = 0;
    size_t              strip_offsets_offset;
    size_t              strip_bytes_count_count = 0;
    size_t              strip_bytes_count_offset;
    int                 planar_configuration = 0;
    int                 channels = 0;
    size_t              bits_per_sample_offset = 0;
    std::vector<int>    bits_per_sample;
    uint32_t            next_offset;
    std::vector<size_t> strip_offsets;
    std::vector<size_t> strip_byte_counts;
private:
    directory_entry();
    static size_t read_value(tiff::data_type type, bstream_base& bs);
};

class nervana::tiff::reader
{
public:
    reader(const char* data, size_t size);
    bool is_tiff();

private:
    bstream_mem     bstream;

    file_header     header;
};
