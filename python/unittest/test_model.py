# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import json


def test_model():
    ark.init()

    input_tensor = ark.tensor([64, 64], ark.fp16)
    other_tensor = ark.tensor([64, 64], ark.fp16)
    ark.add(input_tensor, other_tensor)

    m = ark.Model.get_model().compress()
    m_json = json.loads(m.serialize())

    assert m_json.get("Nodes", None) is not None
    assert len(m_json["Nodes"]) == 1
    assert m_json["Nodes"][0].get("Op", None) is not None
    assert m_json["Nodes"][0]["Op"].get("Type", None) == "Add"

    ark.Model.reset()

    m = ark.Model.get_model().compress()
    m_json = json.loads(m.serialize())

    assert m_json.get("Nodes", None) is not None
    assert len(m_json["Nodes"]) == 0
