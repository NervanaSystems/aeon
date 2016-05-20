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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

 using namespace std;

DataGen::DataGen() :
    _path(),
    _prefix("archive-"),
    _maxItems(4000),
    _maxSize(-1),
    _setSize(100000),
    _imageRows(256),
    _imageCols(256),
    _pathExisted(false)
{
}

DataGen::~DataGen() {
}

DataGen& DataGen::Directory( const std::string& dir ) {
    _path = dir;
    return *this;
}

DataGen& DataGen::Prefix( const std::string& prefix ) {
    _prefix = prefix;
    return *this;
}

DataGen& DataGen::MacrobatchMaxItems( int max ) {
    assert(max>0);
    _maxItems = max;
    return *this;
}

DataGen& DataGen::MacrobatchMaxSize( int max ) {
    assert(max>0);
    _maxSize = max;
    return *this;
}

DataGen& DataGen::DatasetSize( int size ) {
    assert(size>0);
    _setSize = size;
    return *this;
}

DataGen& DataGen::ImageSize( int rows, int cols ) {
    assert(rows>0);
    assert(cols>0);
    _imageRows = rows;
    _imageCols = cols;
    return *this;
}

vector<unsigned char> DataGen::RenderImage( int number, int label ) {
    cv::Mat image = cv::Mat( _imageRows, _imageCols, CV_8UC3 );
    image = cv::Scalar(255,255,255);
    auto fontFace = cv::FONT_HERSHEY_PLAIN;
    string text = to_string(number) + ", " + to_string(label);
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

int DataGen::Create() {
    int rc = -1;
    int fileNo = 0;
    _pathExisted = exists(_path);
    int imageNumber = 0;
    if( _pathExisted || mkdir(_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 ) {
        int remainder = _setSize;
        while(remainder > 0) {
            int batchSize = min(remainder,_maxItems);
            string fileName = _path + "/" + _prefix + to_string(fileNo++) + ".cpio";
            _fileList.push_back(fileName);
            BatchFile bf;
            bf.openForWrite(fileName, "");
            for(int i=0; i<batchSize; i++) {
                int target = 42;
                vector<unsigned char> imageData = RenderImage( imageNumber++, target );
                int imageSize = imageData.size();
                bf.writeItem((char*)imageData.data(),(char*)&target,(uint)imageSize,(uint)sizeof(target));
            }
            bf.close();
            remainder -= batchSize;
        }
    } else {
        cout << "failed to create path " << _path << endl;
    }
    return rc;
}

void DataGen::Delete() {
    for( const string& f : _fileList ) {
        remove(f.c_str());
    }
    if(!_pathExisted) {
        // delete directory
        remove(_path.c_str());
    }
}

bool DataGen::exists(const string& fileName) {
    struct stat stats;
    return stat(fileName.c_str(), &stats) == 0;
}

std::string DataGen::GetDatasetPath() {
    return _path;
}

std::vector<std::string> DataGen::GetFiles() {
    return _fileList;
}

