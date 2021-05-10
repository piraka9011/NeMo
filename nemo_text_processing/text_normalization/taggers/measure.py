# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_ALPHA,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g. 
        -12kg -> measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" }
        1kg -> measure { cardinal { integer: "one" } units: "kilogram" }
        .5kg -> measure { decimal { fractional_part: "five" } units: "kilograms" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")
        cardinal_graph = cardinal.graph

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_plural = convert_space(graph_unit @ SINGULAR_TO_PLURAL)
        graph_unit = convert_space(graph_unit)
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph_unit2 = pynini.cross("/", "per") + delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit

        optional_graph_unit2 = pynini.closure(
            delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2, 0, 1,
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + pynutil.insert("\"")
        )

        unit_singular = (
            pynutil.insert("units: \"") + (graph_unit + optional_graph_unit2 | graph_unit2) + pynutil.insert("\"")
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("1", "one")
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_singular
        )

        optional_alpha = pynini.closure(pynutil.insert(" ") + NEMO_ALPHA)
        optional_serial_end = pynini.closure(pynini.cross('-', '')) + NEMO_ALPHA + pynutil.insert(" ")
        optional_serial_start = pynini.closure(
            (NEMO_ALPHA + pynini.cross('-', ' ')) | (NEMO_ALPHA + pynutil.insert(" "))
        )

        serial_graph_cardinal_end = cardinal.graph + (
            (pynutil.insert(" ") + NEMO_ALPHA) | (pynini.cross('-', ' ') + NEMO_ALPHA)
        )
        serial_graph_cardinal_start = (
            NEMO_ALPHA + (pynutil.insert(" ")) | (pynini.cross('-', ' ') + NEMO_ALPHA)
        ) + cardinal.graph

        # serial_graph_cardinal = optional_serial_start + serial_graph_cardinal + pynini.closure(pynutil.insert(" ") + serial_graph_cardinal)

        serial_graph_decimal = decimal.final_graph_wo_negative + (
            (pynutil.insert(" ") + NEMO_ALPHA) | (pynini.cross('-', ' ') + NEMO_ALPHA)
        )
        serial_graph_decimal = (
            optional_serial_start + serial_graph_decimal + pynini.closure(pynutil.insert(" ") + serial_graph_decimal)
        )

        subgraph_cardinal = pynutil.add_weight(subgraph_cardinal.optimize(), 1.09)
        subgraph_cardinal |= pynutil.add_weight(
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + (serial_graph_cardinal_end | serial_graph_cardinal_start)
            + delete_space
            + pynutil.insert("\" } units: \"serial\""),
            2.1,
        )

        subgraph_decimal = pynutil.add_weight(subgraph_decimal.optimize(), 1.09)
        subgraph_decimal |= pynutil.add_weight(
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + serial_graph_decimal
            + pynutil.insert(" } units: \"\""),
            2.1,
        )

        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
