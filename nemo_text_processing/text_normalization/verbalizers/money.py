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

from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    get_abs_path,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):

    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "twelve" fractional_part: "o five" currency: "dollars" } -> twelve o five dollars

    Args:
        decimal: DecimalFst
    """

    def __init__(self, decimal: GraphFst, deterministic=True):
        super().__init__(name="money", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "minus ") + delete_space, 0, 1)
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_integer = pynini.closure(integer + delete_space + insert_space, 0, 1)
        fractional_default = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        fractional = pynutil.insert("point ") + fractional_default

        if not deterministic:
            fractional2 = (
                pynutil.delete("fractional_part:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.cross('zero', '')
                + pynini.closure(pynini.cross(' zero', ''))
                + delete_space
                + pynutil.delete("\"")
                + delete_space
            )

            fractional = fractional | fractional2 | fractional_default

        quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(quantity, 0, 1)

        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        if deterministic:
            graph = optional_sign + (integer | integer + quantity | optional_integer + fractional + optional_quantity)
            graph = graph + delete_space + pynutil.insert(" ") + unit
        else:
            """
            (' integer_part: "three" MAJOR fractional_part: "eight five" , " read Jess aloud.', 69.33)
            """
            # graph = optional_sign + (integer | integer + quantity | optional_integer + fractional + optional_quantity)
            # graph = optional_integer + fractional + optional_quantity + delete_space + pynutil.insert(" ") + unit + pynutil.insert(' CENTS')

            minor_currencies = []
            with open(get_abs_path("data/currency_minor.tsv"), 'r') as f:
                for line in f:
                    min_cur = line.strip()
                    minor_currencies.append(pynini.closure(pynutil.insert(min_cur + " "), 0, 1))

            graph = (
                integer
                + delete_space
                + insert_space
                + unit
                + delete_space
                + insert_space
                + pynini.union(*minor_currencies)
                + fractional
                + insert_space
                + (pynutil.insert('cents') | pynutil.insert('pence'))
            )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
