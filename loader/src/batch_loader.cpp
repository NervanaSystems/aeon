#include <math.h>

#include "batch_loader.hpp"

uint BatchLoader::blockCount(uint block_size) {
    return ceil((float)objectCount() / (float)block_size);
}
