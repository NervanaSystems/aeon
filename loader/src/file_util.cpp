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

#include <sys/stat.h>
#include <stdexcept>
#include <ftw.h>
#include <unistd.h>
#include <sstream>
#include <string.h>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <fcntl.h>
#include <cassert>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/file.h>

#include "file_util.hpp"

using namespace std;

// maximum number of files opened by nftw file enumeration function
// For some platforms (older linux), OPEN_MAX needs to be defined
#ifndef OPEN_MAX
#define OPEN_MAX 128
#endif

//nervana::temp_file::temp_file()
//{
//}

//nervana::temp_file::~temp_file()
//{
//}

string nervana::file_util::path_join(const string& s1, const string& s2)
{
    string rc;
    if(s2.size() > 0) {
        if(s2[0] == '/') {
            rc = s2;
        } else if(s1.size() > 0) {
            rc = s1;
            if(rc[rc.size()-1] != '/') {
                rc += "/";
            }
            rc += s2;
        } else {
            rc = s2;
        }
    } else {
        rc = s1;
    }
    return rc;
}

off_t nervana::file_util::get_file_size(const string& filename)
{
    // ensure that filename exists and get its size

    struct stat stats;
    if (stat(filename.c_str(), &stats) == -1) {
        throw std::runtime_error("Could not find file: \"" + filename + "\"");
    }

    return stats.st_size;
}

int nervana::file_util::rm(const char *path, const struct stat *s, int flag, struct FTW *f)
{
    // see http://stackoverflow.com/a/1149837/2093984
    // Call unlink or rmdir on the path, as appropriate.
    int status;

    switch(flag) {
        default:     status = unlink(path); break;
        case FTW_DP: status = rmdir (path);
    }

    if(status != 0) {
        stringstream message;
        message << "error deleting file " << path;
        throw std::runtime_error(message.str());
    }

    return status;
}

void nervana::file_util::remove_directory(const string& dir)
{
    // see http://stackoverflow.com/a/1149837/2093984
    // FTW_DEPTH: handle directories after its contents
    // FTW_PHYS: do not follow symbolic links
    if(nftw(dir.c_str(), rm, OPEN_MAX, FTW_DEPTH | FTW_PHYS)) {
        throw std::runtime_error("error deleting directory " + dir);
    }
}

void nervana::file_util::remove_file(const string& file)
{
    remove(file.c_str());
}

bool nervana::file_util::make_directory(const string& dir)
{
    if(mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
        if(errno == EEXIST) {
            // not really an error, the directory already exists
            return false;
        }
        throw std::runtime_error("error making directory " + dir + " " + strerror(errno));
    }
    return true;
}

string nervana::file_util::make_temp_directory(const string& path)
{
    string fname = path.empty() ? file_util::get_temp_directory() : path;
    string tmp_template = file_util::path_join(fname, "aeonXXXXXX");
    char* tmpname = strdup(tmp_template.c_str());

    mkdtemp(tmpname);

    string rc = tmpname;
    free(tmpname);
    return rc;
}

std::string nervana::file_util::get_temp_directory()
{
    const vector<string> potential_tmps = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};

    const char* path;
    for(const string& var : potential_tmps)
    {
        path = getenv(var.c_str());
        if (path != 0) break;
    }
    if (path == 0) path = "/tmp";

    return path;
}

vector<char> nervana::file_util::read_file_contents(const string& path)
{
    ifstream file(path, ios::binary);
    if(!file) {
        throw std::runtime_error("error opening file '" + path + "'");
    }
    vector<char> data((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    return data;
}

void nervana::file_util::iterate_files(const string& path, std::function<void(const string& file)> func)
{
    DIR* dir;
    struct dirent* ent;
    if((dir = opendir(path.c_str())) != NULL) {
        while((ent = readdir(dir)) != NULL) {
            string file = ent->d_name;
            if (file != "." && file != "..") {
                file = file_util::path_join(path, file);
                func(file);
            }
        }
        closedir(dir);
    }
    else {
        throw std::runtime_error("error enumerating file " + path);
    }
}

string nervana::file_util::tmp_filename(const string& extension)
{
    string tmp_template = file_util::path_join(file_util::get_temp_directory(), "tmpfileXXXXXX"+extension);
    char* tmpname = strdup(tmp_template.c_str());

    // mkstemp opens the file with open() so we need to close it
    close(mkstemps(tmpname,extension.size()));

    string rc = tmpname;
    free(tmpname);
    return rc;
}

void nervana::file_util::touch(const std::string& filename)
{
    // inspired by http://chris-sharpe.blogspot.com/2013/05/better-than-systemtouch.html
    int fd = open(
         filename.c_str(), O_WRONLY|O_CREAT|O_NOCTTY|O_NONBLOCK, 0666
    );
    assert(fd>=0);
    close(fd);

    // update timestamp for filename
    int rc = utimes(filename.c_str(), nullptr);
    assert(!rc);
}

bool nervana::file_util::exists(const std::string& filename)
{
    struct stat buffer;
    return (stat (filename.c_str(), &buffer) == 0);
}

int nervana::file_util::try_get_lock(const std::string& filename)
{
    mode_t m = umask(0);
    int fd = open(filename.c_str(), O_RDWR|O_CREAT, 0666);
    umask(m);
    if(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0)
    {
        close(fd);
        fd = -1;
    }
    return fd;
}

void nervana::file_util::release_lock(int fd, const std::string& filename)
{
    if(fd >= 0) {
        remove_file(filename);
        close(fd);
    }
}