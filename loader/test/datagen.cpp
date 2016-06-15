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

#include <iostream>
#include <sys/stat.h>
#include "datagen.hpp"
#include "batchfile.hpp"
#include "util.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

 using namespace std;

DataGen::DataGen() :
    _imageRows(256),
    _imageCols(256)
{
}

DataGen::~DataGen() {
}

DataGen& DataGen::ImageSize( int rows, int cols ) {
    assert(rows>0);
    assert(cols>0);
    _imageRows = rows;
    _imageCols = cols;
    return *this;
}

vector<unsigned char> DataGen::render_datum( int number ) {
    cv::Mat image = cv::Mat( _imageRows, _imageCols, CV_8UC3 );
    image = cv::Scalar(255,255,255);
    auto fontFace = cv::FONT_HERSHEY_PLAIN;
    string text = to_string(number);// + ", " + to_string(label);
    float scale = 2.0 / 256. * _imageRows;
    int thickness = 1;
    int baseline=0;
    cv::Size textSize = getTextSize(text, fontFace, scale, thickness, &baseline);
    baseline += thickness;

    cv::Point position((_imageRows - textSize.width)/2, (_imageCols + textSize.height)/2);

    cv::putText( image, text, position, fontFace, scale, cv::Scalar(0,0,255) );
    vector<unsigned char> result;
    cv::imencode( ".png", image, result );
    return result;
}

vector<unsigned char> DataGen::render_target( int number ) {
    int target = number + 42;
    vector<unsigned char> rc(4);
    nervana::pack_le<int>((char*)&rc[0],target);
    
//    for( int i=0; i<4; i++ ) printf( "0x%02X ", rc[i] );
//    printf( "\n" );
    
    return rc;
}

//int DataGen::Create() {
//    int rc = -1;
//    int fileNo = 0;
//    _pathExisted = exists(_path);
//    int imageNumber = 0;
//    if( _pathExisted || mkdir(_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 ) {
//        int remainder = _setSize;
//        while(remainder > 0) {
//            int batchSize = min(remainder,_maxItems);
//            string fileName = _path + "/" + _prefix + to_string(fileNo++) + ".cpio";
//            _fileList.push_back(fileName);
//            BatchFileWriter bf;
//            bf.open(fileName, "");
//            for(int i=0; i<batchSize; i++) {
//                int target = imageNumber + 42;
//                vector<unsigned char> imageData = RenderImage( imageNumber++, target );
//                int imageSize = imageData.size();
//                bf.writeItem((char*)imageData.data(),(char*)&target,(uint)imageSize,(uint)sizeof(target));
//            }
//            bf.close();
//            remainder -= batchSize;
//        }
//    } else {
//        cout << "failed to create path " << _path << endl;
//    }
//    return rc;
//}

//void DataGen::Delete() {
//    for( const string& f : _fileList ) {
//        remove(f.c_str());
//    }
//    if(!_pathExisted) {
//        // delete directory
//        remove(_path.c_str());
//    }
//}
