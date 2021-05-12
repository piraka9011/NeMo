# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import re
import json
import os
from argparse import ArgumentParser
from typing import List

from nemo_text_processing.text_normalization.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.verbalizers.verbalize_final import VerbalizeFinalFst
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

from nemo.collections import asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

try:
    import pynini

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class NormalizerWithAudio(Normalizer):
    """
    Normalizer class that converts text from written to spoken form. 
    Useful for TTS preprocessing. 

    Args:
        input_case: expected input capitalization
    """

    def __init__(self, input_case: str):
        super().__init__(input_case)

        self.tagger = ClassifyFst(input_case=input_case, deterministic=False)
        self.verbalizer = VerbalizeFinalFst(deterministic=False)
        self.semiotic_classes = ['money', 'cardinal', 'decimal', 'measure', 'date', 'electronic', 'ordinal', 'time']

    def normalize_with_audio(self, text: str, transcript: str, asr_vocabulary: List[str], verbose: bool=False, asr_lower: bool=True) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms
        Args:
            text: string that may include semiotic classes
            transcript: transcription of the audio
            verbose: whether to print intermediate meta information
        Returns: spoken form that matches the audio file best
        """
        def get_tagged_texts(text):
            text = text.strip()
            if not text:
                if verbose:
                    print(text)
                return text
            text = pynini.escape(text)
            tagged_lattice = self.find_tags(text)
            tagged_texts = self.select_all_semiotic_tags(tagged_lattice)
            return tagged_texts

        tagged_texts = set(get_tagged_texts(text) + get_tagged_texts(self.preprocess(text)))
        normalized_texts = []
        for tagged_text in tagged_texts:
            if 'currency' in tagged_text and 'fract' in tagged_text:
                print(tagged_text)
            self.parser(tagged_text)
            tokens = self.parser.parse()
            tags_reordered = self.generate_permutations(tokens)
            for tagged_text in tags_reordered:
                tagged_text = pynini.escape(tagged_text)

                verbalizer_lattice = self.find_verbalizer(tagged_text)
                if verbalizer_lattice.num_states() == 0:
                    continue

                verbalized = self.get_all_verbalizers(verbalizer_lattice)
                # verbalized = [self.select_verbalizer(verbalizer_lattice)]
                for verbalized_option in verbalized:
                    # TODO check this
                    normalized_texts.append(verbalized_option)

        if len(normalized_texts) == 0:
            raise ValueError()

        punctuation = '!,.:;?'
        for i in range(len(normalized_texts)):
            for punct in punctuation:
                normalized_texts[i] = normalized_texts[i].replace(f' {punct}', punct)
            normalized_texts[i] = (
                normalized_texts[i].replace('--', '-').replace('( ', '(').replace(' )', ')').replace('  ', ' ')
            )
        normalized_texts = set(normalized_texts)

        normalized_options = []
        for text in normalized_texts:
            if asr_lower:
                text_clean = text.lower()
            else:
                text_clean = text
            for punct in punctuation:
                text_clean = text_clean.replace(punct, '')
            text_clean = text_clean.replace('-', ' ')
            cer = round(word_error_rate([transcript], [text_clean], use_cer=True) * 100, 2)
            normalized_options.append((text, cer))

        normalized_options = sorted(normalized_options, key=lambda x: x[1])
        if verbose:
            for option in normalized_options:
                print(option)
        return normalized_options[0]

    def select_all_semiotic_tags(self, lattice: 'pynini.FstLike', n=100) -> List[str]:
        tagged_text_options = pynini.shortestpath(lattice, nshortest=n)
        tagged_text_options = [t[1] for t in tagged_text_options.paths("utf8").items()]
        return tagged_text_options

    def get_all_verbalizers(self, lattice: 'pynini.FstLike', n=100) -> List[str]:
        verbalized_options = pynini.shortestpath(lattice, nshortest=n)
        verbalized_options = [t[1] for t in verbalized_options.paths("utf8").items()]
        return verbalized_options

    def preprocess(self, text: str):
        text = text.replace('--', '-')
        space_right = '!?:;,.-()*+-/<=>@^_'
        space_both = '-()*+-/<=>@^_'

        for punct in space_right:
            text = text.replace(punct, punct + ' ')
        for punct in space_both:
            text = text.replace(punct, ' ' + punct + ' ')

        # remove extra space
        text = re.sub(r' +', ' ', text)
        return text


def _get_asr_model(asr_model: nemo_asr.models.EncDecCTCModel):
    if os.path.exists(args.model):
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(asr_model)
    elif args.model in nemo_asr.models.EncDecCTCModel.get_available_model_names():
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(asr_model)
    else:
        raise ValueError(
            f'Provide path to the pretrained checkpoint or choose from {nemo_asr.models.EncDecCTCModel.get_available_model_names()}'
        )
    vocabulary = asr_model.cfg.decoder.vocabulary
    return asr_model, vocabulary

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", help="input string", default=None, type=str
    )
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument("--audio_data", help="path to audio file or .json manifest", required=True)
    parser.add_argument(
        '--model', type=str, default='QuartzNet15x5Base-En', help='Pre-trained model name or path to model checkpoint'
    )
    return parser.parse_args()

def main(args):
    if not os.path.exists(args.audio_data):
        raise ValueError(f'{args.audio_data} not found.')
    else:
        normalizer = NormalizerWithAudio(input_case=args.input_case)

        if 'json' in args.audio_data:
            manifest = args.audio_data
            manifest_out = manifest.replace('.json', '_nemo_wfst.json')
            with open(manifest, 'r') as f:
                with open(manifest_out, 'w') as f_out:
                    for line in tqdm(f):
                        line = json.loads(line)
                        audio = line['audio_filepath']
                        if 'transcript' in line:
                            transcript = line['transcript']
                            # TODO fix
                            asr_vocabulary = None

                        else:
                            asr_model, asr_vocabulary = _get_asr_model(args.model)
                            transcript = asr_model.transcribe([audio])[0]
                        args.input = line['text']
                        normalized_text, cer = normalizer.normalize_with_audio(
                            args.input, transcript, asr_vocabulary, verbose=args.verbose
                        )

                        if cer > line['CER_gt_normalized']:
                            print(f'input     : {args.input}')
                            print(f'transcript: {transcript}')
                            print('gt  :', line['gt_normalized'], line['CER_gt_normalized'])
                            print('wfst:', normalized_text, cer)
                            print('audio:', line['audio_filepath'])
                            print('=' * 40)
                            line['nemo_wfst'] = normalized_text
                            line['CER_nemo_wfst'] = cer
                            f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
            print(f'Normalized version saved at {manifest_out}.')
        else:
            asr_model, _ = _get_asr_model(args.model)
            transcript = asr_model.transcribe([args.audio_data])[0]
            normalized_text, cer = normalizer.normalize_with_audio(args.input, transcript, verbose=args.verbose)
            print(normalized_text)

if __name__ == "__main__":
    args = parse_args()

    if args.input:
        normalizer = NormalizerWithAudio(input_case=args.input_case)
        normalized_text, cer = normalizer.normalize_with_audio(args.input, None, None, verbose=args.verbose)
    else:
        main(args)


