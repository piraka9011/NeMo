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
    NEMO_NOT_SPACE,
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

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic=True):
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

        num_graph = cardinal.graph
        if deterministic:
            num_graph = cardinal.graph | cardinal.single_digits_graph

        subgraph_cardinal = subgraph_cardinal.optimize()

        if not deterministic:
            serial_graph_cardinal_start = (
                    pynini.closure(NEMO_ALPHA + pynutil.insert(" ") | (NEMO_ALPHA + pynini.cross('-', ' ')),
                                   1) + num_graph
            )
            serial_end = pynini.closure(
                pynutil.insert(" ") + NEMO_ALPHA + pynini.closure(pynutil.insert(" ") + num_graph))

            serial_graph_cardinal_end = num_graph + (
                    (pynutil.insert(" ") + NEMO_ALPHA) | (pynini.cross('-', ' ') + NEMO_ALPHA)
            )
            serial_end2 = pynini.closure(
                pynutil.insert(" ")
                + num_graph
                + pynini.closure((pynutil.insert(" ") | pynini.cross("-", " ")) + NEMO_ALPHA)
            )

            serial_graph = (serial_graph_cardinal_start | serial_graph_cardinal_end) + pynini.closure(
                serial_end | serial_end2
            )

            subgraph_cardinal |= (pynutil.insert("cardinal { ")
                + optional_graph_negative
                + pynutil.insert("integer: \"")
                + serial_graph
                + delete_space
                + pynutil.insert("\" } units: \"serial\""),
            )

        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
