// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "file_io.h"
#include "logging.h"

using namespace std;

static bool get_stat(const char *path, struct stat *st)
{
    if (stat(path, st) == -1) {
        switch (errno) {
        case EACCES:
            LOG(ark::ERROR, "permission denied: ", path);
        case EFAULT:
            LOG(ark::ERROR, "bad address: ", path);
        case ENOENT:
        case ENOTDIR:
            return false;
        default:
            LOG(ark::ERROR, "stat() for file path ", path,
                " failed with errno ", errno);
        };
    }
    return true;
}

namespace ark {

bool is_exist(const string &path)
{
    struct stat st;
    return get_stat(path.c_str(), &st);
}

bool is_dir(const string &path)
{
    struct stat st;
    if (!get_stat(path.c_str(), &st)) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

bool is_file(const string &path)
{
    struct stat st;
    if (!get_stat(path.c_str(), &st)) {
        return false;
    }
    return S_ISREG(st.st_mode);
}

int create_dir(const string &path)
{
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        return errno;
    }
    return 0;
}

int remove_dir(const string &path)
{
    if (rmdir(path.c_str()) == -1) {
        return errno;
    }
    return 0;
}

// Helper function to remove all files in a directory given a file descriptor.
int clear_dirat_helper(int dir_fd, const char *name)
{
    int r = -1;
    int fd = openat(dir_fd, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);

    if (fd >= 0) {
        struct stat statbuf;

        if (!fstat(fd, &statbuf)) {
            if (S_ISDIR(statbuf.st_mode)) {
                r = clear_dirat_helper(fd, ".");
                if (!r) {
                    r = unlinkat(dir_fd, name, AT_REMOVEDIR);
                }
            } else {
                r = unlinkat(dir_fd, name, 0);
            }
        }
        close(fd);
    }

    return r;
}

// Remove all files in a directory.
int clear_dir(const string &path)
{
    const char *path_c = path.c_str();
    DIR *d = opendir(path_c);
    int r = -1;

    if (d) {
        struct dirent *p;
        int dir_fd = dirfd(d);

        while ((p = readdir(d))) {
            // Skip the names "." and ".." as we don't want to recurse on
            // them.
            if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
                continue;
            }
            r = clear_dirat_helper(dir_fd, p->d_name);
        }
        closedir(d);
    }

    return r;
}

vector<string> list_dir(const string &path)
{
    string path_str;
    if (path[path.size() - 1] == '/') {
        path_str = path.substr(0, path.size() - 1);
    } else {
        path_str = path;
    }
    const char *path_c = path_str.c_str();
    DIR *d = opendir(path_c);
    size_t path_len = strlen(path_c);

    vector<string> ret;

    if (d) {
        struct dirent *p;

        while ((p = readdir(d))) {
            char *buf;
            size_t len;

            // Skip the names "." and ".." as we don't want to recurse on them.
            if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
                continue;
            }
            len = path_len + strlen(p->d_name) + 2;
            buf = (char *)malloc(len);

            if (buf) {
                struct stat statbuf;

                snprintf(buf, len, "%s/%s", path_c, p->d_name);
                if (!stat(buf, &statbuf)) {
                    ret.emplace_back(buf);
                }
                free(buf);
            }
        }
        closedir(d);
    }

    return ret;
}

string read_file(const string &path)
{
    ifstream file(path);
    stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void write_file(const string &path, const string &data)
{
    ofstream file(path, ios::out | ios::trunc);
    file << data;
}

int remove_file(const string &path)
{
    LOG(DEBUG, "remove file: ", path);
    return remove(path.c_str());
}

string get_dir(const string &path)
{
    size_t len = path.size();
    while (len-- > 0) {
        if (path[len] == '/') {
            break;
        }
    }
    return path.substr(0, len);
}

} // namespace ark
