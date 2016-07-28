#include <math.h>

#include "batch_loader.hpp"

BatchLoader::BatchLoader(uint block_size) :
    _block_size(block_size) {
}

uint BatchLoader::blockSize() {
    return _block_size;
}

uint BatchLoader::blockCount() {
    return ceil((float)objectCount() / (float)_block_size);
}
