// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_json.hpp"

#include <sstream>

#include "logging.h"

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

static void verify_format_json(const std::string &name, const Json &json,
                               const std::vector<std::string> &required_fields,
                               const std::vector<std::string> &array_fields) {
    for (const auto &field : required_fields) {
        if (!json.contains(field)) {
            ERR(NotFoundError,
                name + ": " + field + " not found. Given: " + json.dump());
        }
    }
    for (const auto &field : array_fields) {
        if (!json.at(field).is_array()) {
            ERR(InvalidUsageError,
                name + ": " + field +
                    " is not an array. Given: " + json.dump());
        }
    }
}

static void verify_format_buffer(const Json &json) {
    const std::vector<std::string> required_fields = {"Id", "Rank", "SendTags",
                                                      "RecvTags"};
    const std::vector<std::string> array_fields = {"SendTags", "RecvTags"};
    verify_format_json("BufferJson", json, required_fields, array_fields);
}

static void verify_format_tensor(const Json &json) {
    const std::vector<std::string> required_fields = {
        "Id",      "DataType",    "Shape", "Strides",
        "Offsets", "PaddedShape", "Buffer"};
    const std::vector<std::string> array_fields = {"Shape", "Strides",
                                                   "Offsets", "PaddedShape"};
    verify_format_json("TensorJson", json, required_fields, array_fields);
    verify_format_buffer(json.at("Buffer"));
}

static void verfiy_format_op(const Json &json, bool need_config) {
    std::vector<std::string> required_fields = {
        "Type",         "Name",          "IsVirtual", "ReadTensors",
        "WriteTensors", "ResultTensors", "Args"};
    std::vector<std::string> array_fields = {"ReadTensors", "WriteTensors",
                                             "ResultTensors"};
    if (need_config) {
        required_fields.push_back("Config");
    }
    verify_format_json("OpJson", json, required_fields, array_fields);
    for (const auto &tensor : json.at("ReadTensors")) {
        verify_format_tensor(tensor);
    }
    for (const auto &tensor : json.at("WriteTensors")) {
        verify_format_tensor(tensor);
    }
    for (const auto &tensor : json.at("ResultTensors")) {
        verify_format_tensor(tensor);
    }
}

static void verify_format_node(const Json &json) {
    const std::vector<std::string> required_fields = {"Id", "ProducerNodeIds",
                                                      "ConsumerNodeIds", "Ops"};
    const std::vector<std::string> array_fields = {"ProducerNodeIds",
                                                   "ConsumerNodeIds", "Ops"};
    verify_format_json("NodeJson", json, required_fields, array_fields);
    for (const auto &op : json.at("Ops")) {
        verfiy_format_op(op, false);
    }
}

static void verify_format_model(const Json &json) {
    verify_format_json("ModelJson", json, {"Rank", "WorldSize", "Nodes"},
                       {"Nodes"});
    for (const auto &node : json.at("Nodes")) {
        verify_format_node(node);
    }
}

ModelJson::ModelJson(const Json &json) : Json(json) {
    verify_format_model(*this);
}

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
    dump_pretty_object(*this, "", 5, ss, indent, indent_step) << "\n";
    return ss.str();
}

static void verify_format_task_info(const Json &json) {
    const std::vector<std::string> required_fields = {"Id", "NumWarps",
                                                      "SramBytes", "Ops"};
    const std::vector<std::string> array_fields = {"Ops"};
    verify_format_json("TaskInfoJson", json, required_fields, array_fields);
    for (const auto &op : json.at("Ops")) {
        verfiy_format_op(op, true);
    }
}

static void verify_format_task_group(const Json &json) {
    verify_format_json("TaskGroupJson", json,
                       {"TaskId", "TaskRange", "Granularity"}, {"TaskRange"});
}

static void verify_format_resource_group(const Json &json) {
    const std::vector<std::string> required_fields = {
        "ProcessorRange", "WarpRange", "SramRange", "TaskGroups"};
    verify_format_json("ResourceGroupJson", json, required_fields,
                       required_fields);
    for (const auto &task_group : json.at("TaskGroups")) {
        verify_format_task_group(task_group);
    }
}

static void verify_format_processor_group(const Json &json) {
    const std::vector<std::string> required_fields = {"ProcessorRange",
                                                      "ResourceGroups"};
    verify_format_json("ProcessorGroupJson", json, required_fields,
                       required_fields);
    for (const auto &resource_group : json.at("ResourceGroups")) {
        verify_format_resource_group(resource_group);
    }
}

static void verify_format_plan(const Json &json) {
    const std::vector<std::string> required_fields = {"Rank",
                                                      "WorldSize",
                                                      "Architecture",
                                                      "NumProcessors",
                                                      "NumWarpsPerProcessor",
                                                      "TaskInfos",
                                                      "ProcessorGroups"};
    for (const auto &field : required_fields) {
        if (!json.contains(field)) {
            ERR(NotFoundError, "PlanJson: " + field + " not found");
        }
    }
    if (!json.at("TaskInfos").is_array()) {
        ERR(InvalidUsageError, "PlanJson: TaskInfos is not an array");
    }
    for (const auto &task_info : json.at("TaskInfos")) {
        verify_format_task_info(task_info);
    }
    if (!json.at("ProcessorGroups").is_array()) {
        ERR(InvalidUsageError, "PlanJson: ProcessorGroups is not an array");
    }
    for (const auto &processor_group : json.at("ProcessorGroups")) {
        verify_format_processor_group(processor_group);
    }
}

PlanJson::PlanJson(const Json &json)
    : Json((json != nullptr) ? json
                             : Json{{"Rank", 0},
                                    {"WorldSize", 1},
                                    {"Architecture", "ANY"},
                                    {"NumProcessors", 1},
                                    {"NumWarpsPerProcessor", 1},
                                    {"TaskInfos", Json::array()},
                                    {"ProcessorGroups", Json::array()}}) {
    verify_format_plan(*this);
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
    dump_pretty_item(json.at("Architecture"), "Architecture", ss,
                     indent + indent_step)
        << ",\n";
    dump_pretty_item(json.at("NumProcessors"), "NumProcessors", ss,
                     indent + indent_step)
        << ",\n";
    dump_pretty_item(json.at("NumWarpsPerProcessor"), "NumWarpsPerProcessor",
                     ss, indent + indent_step)
        << ",\n";
    dump_pretty_array(json.at("TaskInfos"), "TaskInfos", 4, ss,
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
