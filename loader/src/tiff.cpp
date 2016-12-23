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

#include <stdexcept>
#include <iostream>
#include <exception>

#include "tiff.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;
using namespace nervana::tiff;

bool nervana::tiff::is_tiff(const char* data, size_t size)
{
    bool rc = true;
    bstream_mem bs{data, size};

    auto byte_order = bs.readU16();
    if(byte_order == 0x4949)
    {
        bs.set_endian(bstream_base::endian_t::LITTLE);
    }
    else if(byte_order == 0x4D4D)
    {
        bs.set_endian(bstream_base::endian_t::BIG);
    }
    else
    {
        rc = false;
    }
    auto file_id = bs.readU16();
    rc &= file_id == 42;
    return rc;
}

file_header::file_header(bstream_base& bs)
{
    byte_order = bs.readU16();
    if(byte_order == 0x4949)
    {
        bs.set_endian(bstream_base::endian_t::LITTLE);
    }
    else if(byte_order == 0x4D4D)
    {
        bs.set_endian(bstream_base::endian_t::BIG);
    }
    else
    {
        throw runtime_error("file not tiff" );
    }
    file_id    = bs.readU16();
    ifd_offset = bs.readU32();
}

size_t directory_entry::read_value(data_type type, bstream_base& bs)
{
    size_t rc;
    if(type == data_type::SHORT) {
        rc = bs.readU16();
        bs.readU16();
    } else {    // must be long
        rc = bs.readU32();
    }
    return rc;
}

directory_entry::directory_entry()
{
}

directory_entry directory_entry::read(bstream_base& bs)
{
    directory_entry rc;
    uint16_t directory_count = bs.readU16();
    cout << __FILE__ << " " << __LINE__ << " " << directory_count          << endl;

    tag_type    tag;
    data_type   type;
    uint32_t    count;
    uint32_t    value_offset;

    for(int i=0; i<directory_count; i++)
    {
        tag   = (tag_type)bs.readU16();
        type  = (data_type)bs.readU16();
        count = bs.readU32();

        cout << "tag " << (tag_type)tag << endl;
        switch(tag)
        {
        case tag_type::ImageWidth:
            rc.image_width = read_value(type, bs);
            break;
        case tag_type::ImageLength:
            rc.image_length = read_value(type, bs);
            break;
        case tag_type::Compression:
            rc.compression = (compression_t)read_value(type, bs);
            cout << "compression " << rc.compression << endl;
            break;
        case tag_type::PhotometricInterpretation:
            rc.photometric = (photometric_t)read_value(type, bs);
            cout << "photometric " << rc.photometric << endl;
            break;
        case tag_type::StripOffsets:
            rc.strip_offsets_count = count;
            rc.strip_offsets_offset = read_value(type, bs);
            break;
        case tag_type::RowsPerStrip:
            cout << __FILE__ << " " << __LINE__ << " " << count << endl;
            value_offset = bs.readU32();
            break;
        case tag_type::StripByteCounts:
            rc.strip_bytes_count_count = count;
            rc.strip_bytes_count_offset = read_value(type, bs);
            break;
        case tag_type::XResolution:
            value_offset = bs.readU32();
            break;
        case tag_type::YResolution:
            value_offset = bs.readU32();
            break;
        case tag_type::ResolutionUnit:
            value_offset = bs.readU32();
            break;
        case tag_type::SamplesPerPixel:
            rc.channels = read_value(type, bs);
            break;
        case tag_type::PlanarConfiguration:
            rc.planar_configuration = read_value(type, bs);
            cout << __FILE__ << " " << __LINE__ << " " << rc.planar_configuration << endl;
            break;
        case tag_type::BitsPerSample:
            rc.bits_per_sample_offset = read_value(type, bs);
            break;
        case tag_type::SampleFormat:
            cout << __FILE__ << " " << __LINE__ << " " << count << endl;
            value_offset = bs.readU32();
            break;
        default:
            value_offset = bs.readU32();
            cout << "unknow tag " << (tag_type)tag << endl;
        }
    }
    rc.next_offset = bs.readU32();

    bs.seek(rc.bits_per_sample_offset);
    for(int i=0; i<rc.channels; i++)
    {
        rc.bits_per_sample.push_back(bs.readU16());
        cout << "sample " << i << " depth " << rc.bits_per_sample.back() << endl;
    }

    bs.seek(rc.strip_offsets_offset);
    for(int i=0; i<rc.strip_offsets_count; i++)
    {
        rc.strip_offsets.push_back(bs.readU32());
//        cout << "strip " << i << " offset " << rc.strip_offsets.back() << endl;
    }

    bs.seek(rc.strip_bytes_count_offset);
    for(int i=0; i<rc.strip_bytes_count_count; i++)
    {
        rc.strip_byte_counts.push_back(bs.readU32());
//        cout << "strip " << i << " byte count " << rc.strip_byte_counts.back() << endl;
    }

//    cout << "next offset " <<    rc.next_offset << endl;
//    cout << "image_width    " << rc.image_width << endl;
//    cout << "image_length   " << rc.image_length << endl;
//    cout << "compression    " << rc.compression << endl;
//    cout << "photometric    " << rc.photometric << endl;

    return rc;
}

reader::reader(const char* data, size_t size) :
    bstream{data, size},
    header{bstream}
{
//    dump(cout, data, 256);
//    cout << __FILE__ << " " << __LINE__ << " " << hex << header.byte_order << dec << endl;
//    cout << __FILE__ << " " << __LINE__ << " " << header.file_id    << endl;
//    cout << __FILE__ << " " << __LINE__ << " " << header.ifd_offset << endl;
    bstream.seek(header.ifd_offset);
    size_t next;
    do
    {
      directory_entry dir = directory_entry::read(bstream);
      cv::Mat img = dir.read_image(bstream);
      cv::imwrite("output_tiff.png", img);
      next = dir.next_offset;
      bstream.seek(next);
    } while(next != 0);
}

cv::Mat directory_entry::read_image(bstream_base& bs) const
{
    int type;
    switch(bits_per_sample[0])
    {
    case 8:
        type = CV_8UC(channels);
        break;
    case 16:
        type = CV_16UC(channels);
        break;
    default:
        throw std::invalid_argument("bits per sample must be 8 or 16");
    }

    cv::Mat rc{(int)image_length, (int)image_width, type};

    uint8_t* data = rc.data;
    for(int i=0; i<strip_offsets.size(); i++)
    {
        bs.seek(strip_offsets[i]);
        data = bs.read(data, strip_byte_counts[i], channels);
    }

    return rc;
}

ostream& nervana::tiff::operator<<(ostream& out, tag_type tag)
{
    switch(tag)
    {
    case tag_type::NewSubfileType             : out << "NewSubfileType";               break;
    case tag_type::SubfileType                : out << "SubfileType";                  break;
    case tag_type::ImageWidth                 : out << "ImageWidth";                   break;
    case tag_type::ImageLength                : out << "ImageLength";                  break;
    case tag_type::BitsPerSample              : out << "BitsPerSample";                break;
    case tag_type::Compression                : out << "Compression";                  break;
    case tag_type::PhotometricInterpretation  : out << "PhotometricInterpretation";    break;
    case tag_type::Threshholding              : out << "Threshholding";                break;
    case tag_type::CellWidth                  : out << "CellWidth";                    break;
    case tag_type::CellLength                 : out << "CellLength";                   break;
    case tag_type::FillOrder                  : out << "FillOrder";                    break;
    case tag_type::DocumentName               : out << "DocumentName";                 break;
    case tag_type::ImageDescription           : out << "ImageDescription";             break;
    case tag_type::Make                       : out << "Make";                         break;
    case tag_type::Model                      : out << "Model";                        break;
    case tag_type::StripOffsets               : out << "StripOffsets";                 break;
    case tag_type::Orientation                : out << "Orientation";                  break;
    case tag_type::SamplesPerPixel            : out << "SamplesPerPixel";              break;
    case tag_type::RowsPerStrip               : out << "RowsPerStrip";                 break;
    case tag_type::StripByteCounts            : out << "StripByteCounts";              break;
    case tag_type::MinSampleValue             : out << "MinSampleValue";               break;
    case tag_type::MaxSampleValue             : out << "MaxSampleValue";               break;
    case tag_type::XResolution                : out << "XResolution";                  break;
    case tag_type::YResolution                : out << "YResolution";                  break;
    case tag_type::PlanarConfiguration        : out << "PlanarConfiguration";          break;
    case tag_type::PageName                   : out << "PageName";                     break;
    case tag_type::XPosition                  : out << "XPosition";                    break;
    case tag_type::YPosition                  : out << "YPosition";                    break;
    case tag_type::FreeOffsets                : out << "FreeOffsets";                  break;
    case tag_type::FreeByteCounts             : out << "FreeByteCounts";               break;
    case tag_type::GrayResponseUnit           : out << "GrayResponseUnit";             break;
    case tag_type::GrayResponseCurve          : out << "GrayResponseCurve";            break;
    case tag_type::T4Options                  : out << "T4Options";                    break;
    case tag_type::T6Options                  : out << "T6Options";                    break;
    case tag_type::ResolutionUnit             : out << "ResolutionUnit";               break;
    case tag_type::PageNumber                 : out << "PageNumber";                   break;
    case tag_type::TransferFunction           : out << "TransferFunction";             break;
    case tag_type::Software                   : out << "Software";                     break;
    case tag_type::DateTime                   : out << "DateTime";                     break;
    case tag_type::Artist                     : out << "Artist";                       break;
    case tag_type::HostComputer               : out << "HostComputer";                 break;
    case tag_type::Predictor                  : out << "Predictor";                    break;
    case tag_type::WhitePoint                 : out << "WhitePoint";                   break;
    case tag_type::PrimaryChromaticities      : out << "PrimaryChromaticities";        break;
    case tag_type::ColorMap                   : out << "ColorMap";                     break;
    case tag_type::HalftoneHints              : out << "HalftoneHints";                break;
    case tag_type::TileWidth                  : out << "TileWidth";                    break;
    case tag_type::TileLength                 : out << "TileLength";                   break;
    case tag_type::TileOffsets                : out << "TileOffsets";                  break;
    case tag_type::TileByteCounts             : out << "TileByteCounts";               break;
    case tag_type::InkSet                     : out << "InkSet";                       break;
    case tag_type::InkNames                   : out << "InkNames";                     break;
    case tag_type::NumberOfInks               : out << "NumberOfInks";                 break;
    case tag_type::DotRange                   : out << "DotRange";                     break;
    case tag_type::TargetPrinter              : out << "TargetPrinter";                break;
    case tag_type::ExtraSamples               : out << "ExtraSamples";                 break;
    case tag_type::SampleFormat               : out << "SampleFormat";                 break;
    case tag_type::SMinSampleValue            : out << "SMinSampleValue";              break;
    case tag_type::SMaxSampleValue            : out << "SMaxSampleValue";              break;
    case tag_type::TransferRange              : out << "TransferRange";                break;
    case tag_type::JPEGProc                   : out << "JPEGProc";                     break;
    case tag_type::JPEGInterchangeFormat      : out << "JPEGInterchangeFormat";        break;
    case tag_type::JPEGInterchangeFormatLngth : out << "JPEGInterchangeFormatLngth";   break;
    case tag_type::JPEGRestartInterval        : out << "JPEGRestartInterval";          break;
    case tag_type::JPEGLosslessPredictors     : out << "JPEGLosslessPredictors";       break;
    case tag_type::JPEGPointTransforms        : out << "JPEGPointTransforms";          break;
    case tag_type::JPEGQTables                : out << "JPEGQTables";                  break;
    case tag_type::JPEGDCTables               : out << "JPEGDCTables";                 break;
    case tag_type::JPEGACTables               : out << "JPEGACTables";                 break;
    case tag_type::YCbCrCoefficients          : out << "YCbCrCoefficients";            break;
    case tag_type::YCbCrSubSampling           : out << "YCbCrSubSampling";             break;
    case tag_type::YCbCrPositioning           : out << "YCbCrPositioning";             break;
    case tag_type::ReferenceBlackWhite        : out << "ReferenceBlackWhite";          break;
    case tag_type::Copyright                  : out << "Copyright";                    break;
    default: out << "unknown tag '" << (int)tag << "'";                                break;
    }
    return out;
}

ostream& nervana::tiff::operator<<(ostream& out, compression_t v)
{
    switch(v)
    {
    case compression_t::Uncompressed : out << "Uncompressed"; break;
    case compression_t::CCITT_1D     : out << "CCITT_1D"; break;
    case compression_t::Group_3_FAX  : out << "Group_3_FAX"; break;
    case compression_t::Group_4_FAX  : out << "Group_4_FAX"; break;
    case compression_t::LZW          : out << "LZW"; break;
    case compression_t::JPEG         : out << "JPEG"; break;
    case compression_t::PackBits     : out << "PackBits"; break;
    default: out << "unknown photometric '" << (int)v << "'"; break;
    }
    return out;
}

ostream& nervana::tiff::operator<<(ostream& out, photometric_t v)
{
    switch(v)
    {
    case photometric_t::WhiteIsZero         : out << "WhiteIsZero"; break;
    case photometric_t::BlackIsZero         : out << "BlackIsZero"; break;
    case photometric_t::RGB                 : out << "RGB"; break;
    case photometric_t::RGB_Palette         : out << "RGB_Palette"; break;
    case photometric_t::Transparency_mask   : out << "Transparency_mask"; break;
    case photometric_t::CMYK                : out << "CMYK"; break;
    case photometric_t::YCbCr               : out << "YCbCr"; break;
    case photometric_t::CIELab              : out << "CIELab"; break;
    default: out << "unknown photometric '" << (int)v << "'"; break;
    }
    return out;
}
