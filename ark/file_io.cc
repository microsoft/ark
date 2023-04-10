// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "ark/file_io.h"
#include "ark/logging.h"

using namespace std;

static bool get_stat(const char *path, struct stat *st)
{
    if (stat(path, st) == -1) {
        switch (errno) {
        case EACCES:
            LOGERR("permission denied: ", path);
        case EFAULT:
            LOGERR("bad address: ", path);
        case ENOENT:
        case ENOTDIR:
            return false;
        default:
            LOGERR("stat() for file path ", path, " failed with errno ", errno);
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

// Remove all files in a directory.
int clear_dir(const string &path)
{
    const char *path_c = path.c_str();
    DIR *d = opendir(path_c);
    size_t path_len = strlen(path_c);
    int r = -1;

    if (d) {
        struct dirent *p;

        r = 0;
        while (!r && (p = readdir(d))) {
            int r2 = -1;
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
                    if (S_ISDIR(statbuf.st_mode)) {
                        r2 = clear_dir(buf);
                        if (!r2) {
                            r2 = rmdir(buf);
                        }
                    } else {
                        r2 = unlink(buf);
                    }
                }
                free(buf);
            }
            r = r2;
        }
        closedir(d);
    }

    return r;
}

vector<string> list_dir(const string &path)
{
    const char *path_c = path.c_str();
    DIR *d = opendir(path_c);
    size_t path_len = strlen(path_c);
    int r = -1;

    vector<string> ret;

    if (d) {
        struct dirent *p;

        r = 0;
        while (!r && (p = readdir(d))) {
            int r2 = -1;
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
            r = r2;
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

void remove_file(const string &path)
{
    remove(path.c_str());
}

string get_dir(const string &path)
{
    unsigned int len = path.size();
    while (len-- >= 0) {
        if (path[len] == '/') {
            break;
        }
    }
    return path.substr(0, len);
}

} // namespace ark
