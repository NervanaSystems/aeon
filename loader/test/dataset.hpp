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

#include <sys/stat.h>
#include <string>
#include <vector>
#include <cassert>
#include <sys/stat.h>
#include <fstream>

#include "cpio.hpp"
#include "file_util.hpp"

template <typename T>
class dataset
{
public:
    dataset()
        : _path()
        , _prefix("archive-")
        , _maxItems(4000)
        , _maxSize(-1)
        , _setSize(100000)
        , _pathExisted(false)
    {
    }

    T& Directory(const std::string& dir)
    {
        _path = dir;
        return *(T*)this;
    }

    T& Prefix(const std::string& prefix)
    {
        _prefix = prefix;
        return *(T*)this;
    }

    T& MacrobatchMaxItems(int max)
    {
        assert(max > 0);
        _maxItems = max;
        return *(T*)this;
    }

    T& MacrobatchMaxSize(int max)
    {
        assert(max > 0);
        _maxSize = max;
        return *(T*)this;
    }

    T& DatasetSize(int size)
    {
        assert(size > 0);
        _setSize = size;
        return *(T*)this;
    }

    int Create()
    {
        int rc          = -1;
        int fileNo      = 0;
        _pathExisted    = exists(_path);
        int datumNumber = 0;
        if (_pathExisted || mkdir(_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0)
        {
            int remainder = _setSize;
            while (remainder > 0)
            {
                int         batchSize = std::min(remainder, _maxItems);
                std::string fileName  = nervana::file_util::path_join(_path, _prefix + std::to_string(fileNo++) + ".cpio");
                _fileList.push_back(fileName);
                std::ofstream f(fileName, std::ostream::binary);
                if (f)
                {
                    nervana::cpio::writer writer{f};
                    for (int i = 0; i < batchSize; i++)
                    {
                        const std::vector<unsigned char> datum = render_datum(datumNumber);
                        writer.write_record_element((char*)datum.data(), datum.size(), 0);

                        const std::vector<unsigned char> target = render_target(datumNumber);
                        writer.write_record_element((char*)target.data(), target.size(), 1);

                        writer.increment_record_count();
                        datumNumber++;
                    }
                    remainder -= batchSize;
                }
            }
        }
        else
        {
            std::cout << "failed to create path " << _path << std::endl;
        }
        return rc;
    }

    void Delete()
    {
        for (const std::string& f : _fileList)
        {
            remove(f.c_str());
        }
        if (!_pathExisted)
        {
            // delete directory
            remove(_path.c_str());
        }
    }

    std::string              GetDatasetPath() { return _path; }
    std::vector<std::string> GetFiles() { return _fileList; }
protected:
    virtual std::vector<unsigned char> render_target(int datumNumber) = 0;
    virtual std::vector<unsigned char> render_datum(int datumNumber)  = 0;

private:
    bool exists(const std::string& fileName)
    {
        struct stat stats;
        return stat(fileName.c_str(), &stats) == 0;
    }

    std::string _path;
    std::string _prefix;
    int         _maxItems;
    int         _maxSize;
    int         _setSize;
    bool        _pathExisted;

    std::vector<std::string> _fileList;
};
