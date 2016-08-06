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

#include <errno.h>
#include <dirent.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <ftw.h>

#include "cpio.hpp"
#include "batch_loader_cpio_cache.hpp"

using namespace std;

// maximum number of files opened by nftw file enumeration function
#define OPEN_MAX 128

BatchLoaderCPIOCache::BatchLoaderCPIOCache(const string& rootCacheDir,
                                           const string& hash,
                                           const string& version,
                                           shared_ptr<BatchLoader> loader)
: BatchLoader(loader->blockSize()), _loader(loader) {
    invalidateOldCache(rootCacheDir, hash, version);

    _cacheDir = rootCacheDir + "/" + hash + "_" + version;

    makeDirectory(_cacheDir);
}

void BatchLoaderCPIOCache::loadBlock(buffer_in_array& dest, uint block_num) {
    if(loadBlockFromCache(dest, block_num)) {
        return;
    } else {
        _loader->loadBlock(dest, block_num);

        try {
            writeBlockToCache(dest, block_num);
        } catch (std::exception& e) {
            // failure to write block to cache doesn't stop execution, only print an error
            cerr << "ERROR writing block to cache: " << e.what() << endl;
        }
    }
}

bool BatchLoaderCPIOCache::loadBlockFromCache(buffer_in_array& dest, uint block_num) {
    // load a block from cpio cache into dest.  If file doesn't exist,
    // return false.  If loading from cpio cache was successful return
    // true.
    CPIOFileReader reader;

    if(!reader.open(blockFilename(block_num))) {
        // couldn't load the file
        return false;
    }
    // load cpio file into dest one item at a time
    for(int i=0; i < reader.itemCount(); ++i) {
        for (auto d : dest) {
            try {
                reader.read(*d);
            } catch (std::exception& e) {
                d->addException(std::current_exception());
            }
        }
    }

    reader.close();

    // cpio file was read successfully, no need to hit primary data
    // source
    return true;
}

void BatchLoaderCPIOCache::writeBlockToCache(buffer_in_array& buff, uint block_num) {
    CPIOFileWriter writer;
    writer.open(blockFilename(block_num));
    writer.write_all_records(buff);
    writer.close();
}

void BatchLoaderCPIOCache::invalidateOldCache(const string& rootCacheDir,
                                              const string& hash,
                                              const string& version) {
    // remove cache directories that match rootCacheDir and hash but not version

    DIR *dir;
    struct dirent *ent;
    if((dir = opendir(rootCacheDir.c_str())) != NULL) {
        while((ent = readdir(dir)) != NULL) {
            if(filenameHoldsInvalidCache(ent->d_name, hash, version)) {
                removeDirectory(rootCacheDir + "/" + ent->d_name);
            }
        }
        closedir(dir);
    }
    else {
        stringstream message;
        message << "error enumerating old cache in " << rootCacheDir;
        throw std::runtime_error(message.str());
    }
}

bool BatchLoaderCPIOCache::filenameHoldsInvalidCache(const string& filename,
                                                     const string& hash,
                                                     const string& version) {
    // in order for `filename` to hold invalid cache, it must begin with
    // `hash`, but not contain `version`

    if(filename.find(hash) != 0) {
        // filename doesn't start with hash, dont remove it
        return false;
    }
    if(filename.find(version) == string::npos) {
        // filename does start with hash, but doesnt have version, invalidate
        return true;
    }
    // filename does start with hash and does have version, keep, its valid
    return false;
}

int BatchLoaderCPIOCache::rm(const char *path, const struct stat *s, int flag, struct FTW *f) {
    // see http://stackoverflow.com/a/1149837/2093984
    // Call unlink or rmdir on the path, as appropriate.
    int (*rm_func)(const char *);

    switch(flag) {
        default:     rm_func = unlink; break;
        case FTW_DP: rm_func = rmdir;
    }

    int status = rm_func(path);
    if(status != 0) {
        stringstream message;
        message << "error deleting file " << path;
        throw std::runtime_error(message.str());
    }

    return status;
}

void BatchLoaderCPIOCache::removeDirectory(const string& dir) {
    // see http://stackoverflow.com/a/1149837/2093984
    // FTW_DEPTH: handle directories after its contents
    // FTW_PHYS: do not follow symbolic links
    if(nftw(dir.c_str(), rm, OPEN_MAX, FTW_DEPTH | FTW_PHYS)) {
        stringstream message;
        message << "error deleting directory " << dir;
        throw std::runtime_error(message.str());
    }
}

void BatchLoaderCPIOCache::makeDirectory(const string& dir) {
    if(mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
        if(errno == EEXIST) {
            // not really an error, the directory already exists
            return;
        }
        stringstream message;
        message << "error making directory " << dir;
        message << " " << strerror(errno);
        throw std::runtime_error(message.str());
    }
}

string BatchLoaderCPIOCache::blockFilename(uint block_num) {
    stringstream s;
    s << _cacheDir << "/" << block_num << "-" << _block_size << ".cpio";
    return s.str();
}

uint BatchLoaderCPIOCache::objectCount() {
    return _loader->objectCount();
}
