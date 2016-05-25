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

#include <string>
#include <vector>
#include <memory>

class DataGen {
public:
    DataGen();
    ~DataGen();

    DataGen& Directory( const std::string& dir );
    DataGen& Prefix( const std::string& prefix );
    DataGen& MacrobatchMaxItems( int max );
    DataGen& MacrobatchMaxSize( int max );
    DataGen& DatasetSize( int size );
    DataGen& ImageSize( int rows, int cols );
    int Create();
    void Delete();

    std::string GetDatasetPath();
    std::vector<std::string> GetFiles();

private:
    std::string _path;
    std::string _prefix;
    int         _maxItems;
    int         _maxSize;
    int         _setSize;
    int         _imageRows;
    int         _imageCols;
    bool        _pathExisted;

    std::vector<std::string> _fileList;

    bool exists(const std::string& fileName);
    std::vector<unsigned char> RenderImage( int number, int label );
};
