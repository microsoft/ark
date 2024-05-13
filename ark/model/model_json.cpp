// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_json.hpp"

#include <sstream>

static std::stringstream &idnt(std::stringstream &ss, int indent) {
    for (int i = 0; i < indent; ++i) ss << " ";
    return ss;
}

static std::stringstream &dquote(std::stringstream &ss,
                                 const std::string &str) {
    ss << "\"" << str << "\"";
    return ss;
}

namespace ark {

static std::stringstream &dump_pretty_item(const Json &json,
                                           const std::string &key,
                                           std::stringstream &ss, int indent) {
    idnt(ss, indent);
    if (!key.empty()) {
        dquote(ss, key) << ": ";
    }
    ss << json.dump();
    return ss;
}

static std::stringstream &dump_pretty_object(const Json &json,
                                             const std::string &key, int level,
                                             std::stringstream &ss, int indent,
                                             int indent_step);

static std::stringstream &dump_pretty_array(const Json &json,
                                            const std::string &key, int level,
                                            std::stringstream &ss, int indent,
                                            int indent_step) {
    size_t num_item = json.size();
    if (num_item == 0) {
        idnt(ss, indent);
        if (key.empty()) {
            ss << "[]";
        } else {
            dquote(ss, key) << ": []";
        }
        return ss;
    }
    idnt(ss, indent);
    if (key.empty()) {
        ss << "[\n";
    } else {
        dquote(ss, key) << ": [\n";
    }
    for (auto &item : json) {
        bool is_obj_or_array = item.is_object() || item.is_array();
        bool is_number_array =
            item.is_array() && item.size() > 0 && item.at(0).is_number();
        if (level <= 0 || !is_obj_or_array || is_number_array) {
            // last level
            dump_pretty_item(item, "", ss, indent + indent_step);
        } else if (item.is_object()) {
            dump_pretty_object(item, "", level - 1, ss, indent + indent_step,
                               indent_step);
        } else {
            dump_pretty_array(item, "", level - 1, ss, indent + indent_step,
                              indent_step);
        }
        num_item--;
        if (num_item == 0) {
            ss << "\n";
        } else {
            ss << ",\n";
        }
    }
    idnt(ss, indent) << "]";
    return ss;
}

static std::stringstream &dump_pretty_object(const Json &json,
                                             const std::string &key, int level,
                                             std::stringstream &ss, int indent,
                                             int indent_step) {
    size_t num_item = json.size();
    if (num_item == 0) {
        idnt(ss, indent);
        if (key.empty()) {
            ss << "{}";
        } else {
            dquote(ss, key) << ": {}";
        }
        return ss;
    }
    idnt(ss, indent);
    if (key.empty()) {
        ss << "{\n";
    } else {
        dquote(ss, key) << ": {\n";
    }
    for (auto &item : json.items()) {
        bool is_obj_or_array =
            item.value().is_object() || item.value().is_array();
        bool is_number_array = item.value().is_array() &&
                               item.value().size() > 0 &&
                               item.value().at(0).is_number();
        if (level <= 0 || !is_obj_or_array || is_number_array) {
            // last level
            dump_pretty_item(item.value(), item.key(), ss,
                             indent + indent_step);
        } else if (item.value().is_object()) {
            dump_pretty_object(item.value(), item.key(), level - 1, ss,
                               indent + indent_step, indent_step);
        } else {
            dump_pretty_array(item.value(), item.key(), level - 1, ss,
                              indent + indent_step, indent_step);
        }
        num_item--;
        if (num_item == 0) {
            ss << "\n";
        } else {
            ss << ",\n";
        }
    }
    idnt(ss, indent) << "}";
    return ss;
}

std::string ModelJson::dump_pretty(int indent, int indent_step) const {
    std::stringstream ss;
    dump_pretty_object(*this, "", 6, ss, indent, indent_step) << "\n";
    return ss.str();
}

static std::stringstream &dump_pretty_plan(const Json &json,
                                           std::stringstream &ss, int indent,
                                           int indent_step) {
    ss << "{\n";
    dump_pretty_item(json.at("Rank"), "Rank", ss, indent + indent_step)
        << ",\n";
    dump_pretty_item(json.at("WorldSize"), "WorldSize", ss,
                     indent + indent_step)
        << ",\n";
    dump_pretty_item(json.at("NumProcessors"), "NumProcessors", ss,
                     indent + indent_step)
        << ",\n";
    dump_pretty_item(json.at("NumWarpsPerProcessor"), "NumWarpsPerProcessor",
                     ss, indent + indent_step)
        << ",\n";
    dump_pretty_array(json.at("TaskInfos"), "TaskInfos", 5, ss,
                      indent + indent_step, indent_step)
        << ",\n";
    dump_pretty_array(json.at("ProcessorGroups"), "ProcessorGroups", 4, ss,
                      indent + indent_step, indent_step)
        << "\n";
    ss << "}\n";
    return ss;
}

std::string PlanJson::dump_pretty(int indent, int indent_step) const {
    std::stringstream ss;
    dump_pretty_plan(*this, ss, indent, indent_step);
    return ss.str();
}

}  // namespace ark
