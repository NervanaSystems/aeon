#include "batchfile.hpp"

// Some utilities that would be used by batch writers
int readFileLines(const string &filn, LineList &ll) {
    std::ifstream ifs(filn);
    if (ifs) {
        for (string line; std::getline( ifs, line ); /**/ )
           ll.push_back( line );
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}

int readFileBytes(const string &filn, ByteVect &b) {
/* Reads in the binary file as a sequence of bytes, resizing
 * the provided byte vector to fit
*/
    std::ifstream ifs(filn, std::ifstream::binary);
    if (ifs) {
        ifs.seekg (0, ifs.end);
        int length = ifs.tellg();
        ifs.seekg (0, ifs.beg);

        b.resize(length);
        ifs.read(&b[0], length);
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}