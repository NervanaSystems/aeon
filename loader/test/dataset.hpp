#ifndef DATASET_H
#define DATASET_H

#include <sys/stat.h>
#include <string>
#include <vector>
#include <cassert>

#include "batchfile.hpp"

template<typename T>
class dataset
{
public:
    dataset() :
        _path(),
        _prefix("archive-"),
        _maxItems(4000),
        _maxSize(-1),
        _setSize(100000),
        _pathExisted(false)
    {
    
    }

    T& Directory( const std::string& dir ) {
        _path = dir;
        return *(T*)this;
    }
    
    T& Prefix( const std::string& prefix ) {
        _prefix = prefix;
        return *(T*)this;
    }
    
    T& MacrobatchMaxItems( int max ) {
        assert(max>0);
        _maxItems = max;
        return *(T*)this;
    }
    
    T& MacrobatchMaxSize( int max ) {
        assert(max>0);
        _maxSize = max;
        return *(T*)this;
    }
    
    T& DatasetSize( int size ) {
        assert(size>0);
        _setSize = size;
        return *(T*)this;
    }
    
    int Create() {
        int rc = -1;
        int fileNo = 0;
        _pathExisted = exists(_path);
        int datumNumber = 0;
        if( _pathExisted || mkdir(_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 ) {
            int remainder = _setSize;
            while(remainder > 0) {
                int batchSize = min(remainder,_maxItems);
                std::string fileName = _path + "/" + _prefix + std::to_string(fileNo++) + ".cpio";
                _fileList.push_back(fileName);
                BatchFileWriter bf;
                bf.open(fileName, "");
                for(int i=0; i<batchSize; i++) {
                    std::vector<unsigned char> target = render_target( datumNumber );
                    std::vector<unsigned char> datum = render_datum( datumNumber );
                    bf.writeItem((char*)datum.data(),(char*)target.data(),(uint)datum.size(),(uint)target.size());
                    datumNumber++;
                }
                bf.close();
                remainder -= batchSize;
            }
        } else {
            std::cout << "failed to create path " << _path << std::endl;
        }
        return rc;
    }
    
    void Delete() {
        for( const string& f : _fileList ) {
            remove(f.c_str());
        }
        if(!_pathExisted) {
            // delete directory
            remove(_path.c_str());
        }
    }
    
    std::string GetDatasetPath() {
        return _path;
    }
    
    std::vector<std::string> GetFiles() {
        return _fileList;
    }


protected:
    virtual std::vector<unsigned char> render_target( int datumNumber ) = 0;
    virtual std::vector<unsigned char> render_datum( int datumNumber ) = 0;

private:
    bool exists(const std::string& fileName) {
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

#endif // DATASET_H
