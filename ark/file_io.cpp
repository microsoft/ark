// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "file_io.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "logging.hpp"

namespace fs = std::filesystem;

namespace ark {

bool is_exist(const std::string &path) {
    return fs::directory_entry{path}.exists();
}

bool is_dir(const std::string &path) {
    return fs::is_directory(fs::status(path));
}

bool is_file(const std::string &path) {
    return fs::is_regular_file(fs::status(path));
}

int create_dir(const std::string &path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    return ec.value();
}

int remove_dir(const std::string &path) {
    LOG(DEBUG, "remove dir: ", path);
    std::error_code ec;
    fs::remove_all(path, ec);
    return ec.value();
}

// Remove all files in a directory.
int clear_dir(const std::string &path) {
    LOG(DEBUG, "clear dir: ", path);
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(path, ec)) {
        if (ec.value() != 0) {
            return ec.value();
        }
        fs::remove_all(entry.path(), ec);
        if (ec.value() != 0) {
            return ec.value();
        }
    }
    return ec.value();
}

std::vector<std::string> list_dir(const std::string &path) {
    std::vector<std::string> files;
    for (const auto &entry : fs::directory_iterator(path)) {
        files.push_back(entry.path().string());
    }
    return files;
}

std::string read_file(const std::string &path) {
    std::ifstream file(path);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void write_file(const std::string &path, const std::string &data) {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    file << data;
}

int remove_file(const std::string &path) {
    LOG(DEBUG, "remove file: ", path);
    std::error_code ec;
    fs::remove(path, ec);
    return ec.value();
}

std::string get_dir(const std::string &path) {
    return fs::path(path).parent_path().string();
}

}  // namespace ark
