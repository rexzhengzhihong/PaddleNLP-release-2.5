# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team.
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
import collections
import logging
import os
from typing import List

from .. import PretrainedTokenizer, BasicTokenizer, WordpieceTokenizer
from ..tokenizer_utils import Trie

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"prophetnet-large-uncased": 512}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def create_trie(unique_no_split_tokens):
    trie = Trie()
    for token in unique_no_split_tokens:
        trie.add(token)
    return trie


class ProphetNetTokenizer(PretrainedTokenizer):
    r"""
    Construct a ProphetNetTokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        x_sep_token (`str`, *optional*, defaults to `"[X_SEP]"`):
            Special second separator token, which can be generated by
            [`ProphetNetForConditionalGeneration`]. It is used to separate bullet-point like
            sentences in summarization, *e.g.*.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    resource_files_names = {"vocab_file": "prophetnet.tokenizer"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "prophetnet-large-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/prophetnet/prophetnet.tokenizer",
        }
    }
    pretrained_init_configuration = {
        "prophetnet-large-uncased": {"do_lower_case": True},
    }
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        bos_token="[SEP]",
        eos_token="[SEP]",
        cls_token="[CLS]",
        x_sep_token="[X_SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        **kwargs
    ):
        self.unique_no_split_tokens = [
            x_sep_token,
            unk_token,
            sep_token,
            bos_token,
            eos_token,
            cls_token,
            pad_token,
            mask_token,
        ]
        self.tokens_trie = create_trie(self.unique_no_split_tokens)
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def tokenize(self, text):
        return self._tokenize(text)

    def _tokenize(self, text):
        """
        Converts a string to a list of tokens.

        Args:
            text (str): The text to be tokenized.

        Returns:
            List[str]: A list of string representing converted tokens.
        """
        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        for i, token in enumerate(tokens):
            if token in no_split_token:
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                # We strip left and right by default
                if right:
                    tokens[i + 1] = right.lstrip()
                if left:
                    tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize_function(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize_function(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a sequence of tokens into ids using the `vocab` attribute (an
        instance of `Vocab`). Override it if needed.

        Args：
            tokens (list[int]): List of token ids.

        Returns:
            list: Converted id list.
        """
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Converts a single index or a sequence of indices to a token or
        a sequence of tokens, using the vocabulary and added tokens.

        Args:
            ids (int or List[int]):
                The token id (or token ids) to be converted to token(s).
            skip_special_tokens (bool, optional):
                Whether or not to remove special tokens in the decoding.
                Defaults to `False` and we do not remove special tokens.

        Returns:
            str or List[str]: The decoded token(s).
        """
        if not isinstance(ids, (list, tuple)):
            return self._convert_id_to_token(ids)
        tokens = [self._convert_id_to_token(_id) for _id in ids]
        if skip_special_tokens:
            return [token for token in tokens if token not in self.all_special_tokens]
        return tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ProphetNet
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        sep = [self.sep_token_id]
        return token_ids_0 + sep + token_ids_1 + sep

    def save_vocabulary(self, save_directory):
        index = 0
        vocab_file = os.path.join(save_directory, self.resource_files_names["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logging.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
