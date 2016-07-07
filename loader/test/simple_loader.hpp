#ifndef SIMPLE_LOADER_HPP
#define SIMPLE_LOADER_HPP

#include <string>

#include "buffer.hpp"
#include "batch_iterator.hpp"

class simple_loader : public BatchIterator
{
public:
    simple_loader(const std::string& repoDir);

    int start() { return 0; }
    void stop() {}

    void read(BufferArray& dest) override;
    void reset() override;

private:
    std::string     _repo;

};

#endif // SIMPLE_LOADER_HPP
