// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"
#include "file_io.h"
#include "include/ark.h"
#include "unittest/unittest_utils.h"
#include <fstream>
#include <algorithm>

ark::unittest::State test_is_exist()
{
    UNITTEST_EQ(ark::is_exist("/"), true);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_is_dir()
{
    UNITTEST_EQ(ark::is_dir("/"), true);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_is_file()
{
    UNITTEST_EQ(ark::is_file("/"), false);
    UNITTEST_EQ(ark::is_file(__FILE__), true);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_create_remove_dir()
{
    std::string tmp_dir = ark::get_env().path_tmp_dir;
    if (!ark::is_exist(tmp_dir)) {
        UNITTEST_EQ(ark::create_dir(tmp_dir), 0);
        UNITTEST_EQ(ark::is_exist(tmp_dir), true);
        UNITTEST_EQ(ark::is_dir(tmp_dir), true);
    }

    auto test_dir = tmp_dir + "/test";
    UNITTEST_EQ(ark::create_dir(test_dir), 0);
    UNITTEST_EQ(ark::is_exist(test_dir), true);
    UNITTEST_EQ(ark::is_dir(test_dir), true);

    UNITTEST_EQ(ark::remove_dir(test_dir), 0);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_list_get_clear_dir()
{
    std::string tmp_dir = ark::get_env().path_tmp_dir;
    if (!ark::is_exist(tmp_dir)) {
        UNITTEST_EQ(ark::create_dir(tmp_dir), 0);
        UNITTEST_EQ(ark::is_exist(tmp_dir), true);
        UNITTEST_EQ(ark::is_dir(tmp_dir), true);
    }

    auto test_file_1 = tmp_dir + "/test1.txt";
    std::ofstream ofs(test_file_1);
    ofs << "test";
    ofs.close();

    auto test_file_2 = tmp_dir + "/test2.txt";
    ofs.open(test_file_2);
    ofs << "test";
    ofs.close();

    UNITTEST_EQ(ark::is_exist(test_file_1), true);
    UNITTEST_EQ(ark::is_file(test_file_1), true);

    UNITTEST_EQ(ark::is_exist(test_file_2), true);
    UNITTEST_EQ(ark::is_file(test_file_2), true);

    auto files = ark::list_dir(tmp_dir);
    UNITTEST_EQ(files.size(), 2UL);
    std::sort(files.begin(), files.end());
    UNITTEST_EQ(files[0], test_file_1);
    UNITTEST_EQ(files[1], test_file_2);

    UNITTEST_EQ(ark::get_dir(test_file_1), tmp_dir);
    UNITTEST_EQ(ark::get_dir(test_file_2), tmp_dir);

    UNITTEST_EQ(ark::clear_dir(tmp_dir), 0);

    UNITTEST_EQ(ark::is_exist(test_file_1), false);
    UNITTEST_EQ(ark::is_exist(test_file_2), false);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_read_write_file()
{
    std::string tmp_dir = ark::get_env().path_tmp_dir;
    if (!ark::is_exist(tmp_dir)) {
        UNITTEST_EQ(ark::create_dir(tmp_dir), 0);
        UNITTEST_EQ(ark::is_exist(tmp_dir), true);
        UNITTEST_EQ(ark::is_dir(tmp_dir), true);
    }

    auto test_file = tmp_dir + "/test.txt";
    std::string data = "test";
    ark::write_file(test_file, data);
    UNITTEST_EQ(ark::is_exist(test_file), true);
    UNITTEST_EQ(ark::is_file(test_file), true);

    auto read_data = ark::read_file(test_file);
    UNITTEST_EQ(read_data, data);

    UNITTEST_EQ(ark::remove_file(test_file), 0);
    UNITTEST_EQ(ark::is_exist(test_file), false);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_is_exist);
    UNITTEST(test_is_dir);
    UNITTEST(test_is_file);
    UNITTEST(test_create_remove_dir);
    UNITTEST(test_list_get_clear_dir);
    UNITTEST(test_read_write_file);
    return 0;
}
