// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_FILE_IO_H_
#define ARK_FILE_IO_H_

#include <string>
#include <vector>

namespace ark {

bool is_exist(const std::string &path);

bool is_dir(const std::string &path);
bool is_file(const std::string &path);
int create_dir(const std::string &path);
int clear_dir(const std::string &path);
std::vector<std::string> list_dir(const std::string &path);

std::string read_file(const std::string &path);
void write_file(const std::string &path, const std::string &data);
void remove_file(const std::string &path);
std::string get_dir(const std::string &path);

} // namespace ark

#endif // ARK_FILE_IO_H_
