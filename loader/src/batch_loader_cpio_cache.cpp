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

#include "batchfile.hpp"
#include "batch_loader_cpio_cache.hpp"

BatchLoaderCPIOCache::BatchLoaderCPIOCache(const char* cacheDir, BatchLoader* loader)
    : _cacheDir(cacheDir), _loader(loader) {
}

void BatchLoaderCPIOCache::loadBlock(BufferPair& dest, uint block_num, uint block_size) {
    if(loadBlockFromCache(dest, block_num, block_size)) {
        return;
    } else {
        cout << 'loadBlock' << block_num << " " << block_size << endl;
        _loader->loadBlock(dest, block_num, block_size);
        writeBlockToCache(dest, block_num, block_size);
    }
}

bool BatchLoaderCPIOCache::loadBlockFromCache(BufferPair& dest, uint block_num, uint block_size) {
    // load a block from cpio cache into dest.  If file doesn't exist,
    // return false.  If loading from cpio cache was successful return
    // true.
    BatchFileReader reader;
    
    if(!reader.tryOpen(blockFilename(block_num, block_size))) {
        // couldn't load the file
        return false;
    }

    // load cpio file into dest one item at a time
    for(int i=0; i < reader.itemCount(); ++i) {
        reader.readToBuffer(*dest.first);
        reader.readToBuffer(*dest.second);
    }

    reader.close();

    // cpio file was read successfully, no need to hit primary data
    // source
    return true;
}

void BatchLoaderCPIOCache::writeBlockToCache(BufferPair& buff, uint block_num, uint block_size) {
    // TODO: why do we have to pass in a dataType if its always empty?
    BatchFileWriter bfw(blockFilename(block_num, block_size), "");

    // would be nice if this was taken care of the BufferPair
    assert(buff.first->getItemCount() == buff.second->getItemCount());

    for(int i=0; i < buff.first->getItemCount(); ++i) {
        // TODO: standardize on name object/datum
        // TODO: standardize on size type int returned from getItem but
        // uint desired from writeItem
        int datum_len;
        char* datum = buff.first->getItem(i, datum_len);
        int target_len;
        char* target = buff.second->getItem(i, target_len);

        bfw.writeItem(datum, target, datum_len, target_len);
    }

    bfw.close();
}

string BatchLoaderCPIOCache::blockFilename(uint block_num, uint block_size) {
    stringstream s;
    s << _cacheDir << block_num << "-" << block_size << ".cpio";
    return s.str();
}
