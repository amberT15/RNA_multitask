import os
from transformers.tokenization_utils import AddedToken,PreTrainedTokenizer
from transformers.utils import is_tf_available, is_torch_available
import collections
import unicodedata
import logging
import itertools
import re
import torch

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
PRETRAINED_VOCAB_FILES_MAP = {"vocab_file": {"dna3": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-3/vocab.txt",
                                             "dna4": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-4/vocab.txt",
                                             "dna5": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-5/vocab.txt",
                                             "dna6": "https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/vocab.txt"}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
                                          "dna3": 512,
                                          "dna4": 512,
                                          "dna5": 512,
                                          "dna6": 512}
PRETRAINED_INIT_CONFIGURATION = {
    "dna3": {"do_lower_case": False},
    "dna4": {"do_lower_case": False},
    "dna5": {"do_lower_case": False},
    "dna6": {"do_lower_case": False}}
VOCAB_KMER = {
    "69": "3",
    "261": "4",
    "1029": "5",
    "4101": "6",}

logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class DNATokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        max_len = 512,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        
        super().__init__(
            max_len = 512,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.max_len = max_len 
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.kmer = VOCAB_KMER[str(len(self.vocab))]

    def prepare_for_tokenization(self, text, **kwargs):
        """ Performs any necessary transformations before tokenization """
        return text
    
    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            text: The sequence to be encoded.
            add_prefix_space: Only applies to GPT-2 and RoBERTa tokenizers. When `True`, this ensures that the sequence
                begins with an empty space. False by default except for when using RoBERTa with `add_special_tokens=True`.
            **kwargs: passed to the `prepare_for_tokenization` preprocessing method.
        """
        all_special_tokens = self.all_special_tokens
        text = self.prepare_for_tokenization(text, **kwargs)

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.all_special_tokens else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text):
        text = self._clean_text(text)
        split_tokens = text.strip().split()
        
        return split_tokens       

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def encode_plus(self,
        text,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation="longest_first",
        padding=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        **kwargs):

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        context_split =  kwargs.get('context_split',False)
        if context_split == False:
            return self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=padding,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
            )
        elif context_split == True:
            pair =( second_ids != None)
            encoded_inputs =  self.prep_context_split(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length ,
                pad_to_max_length=padding,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
            )
            return encoded_inputs

    def prep_context_split(self,
        ids,
        pair_ids=None,
        max_length=None,
        add_special_tokens=True,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        ):

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0
        encoded_inputs={}

        # Handle max sequence length
        if max_length == None:  max_length = self.max_len
        max_length = max_length - self.num_special_tokens_to_add(pair=pair) 
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids = [ids[i:min([i+max_length,len_ids])] for i in range(0, len_ids, max_length)]
            pair_ids = ([pair_ids[i:min([i+max_length,len_pair_ids])] for i in range(0, len_pair_ids, max_length)] if pair else None)
        else:
            ids = [ids]
            pair_ids = [pair_ids]
        seq_list = []
        token_list = []
        type_list = []
        attention_list = []
        for i in range(0,len(ids)):
            i_seq = ids[i]
            pair_seq = pair_ids[i] if pair else None
            
            #Handle special_tokens 
            if add_special_tokens:
                sequence = self.build_inputs_with_special_tokens(i_seq, pair_seq)
                token_type_ids = self.create_token_type_ids_from_sequences(i_seq, pair_seq)
            else: 
                sequence = i_seq + pair_seq if pair else i_seq
                token_type_ids = [0] * len(i_seq) + ([1] * len(pair_seq) if pair else [])
            
            #Handle token mask
            if return_special_tokens_mask:
                token_mask = self.get_special_tokens_mask(i_seq, pair_seq)
                           
            #Handle attention mask
            needs_to_be_padded = pad_to_max_length and (
                max_length
                and len(sequence) < max_length
                or max_length is None
                and len(sequence) < self.max_len
                and self.max_len <= 10000
            )
            if needs_to_be_padded:
                difference = (max_length if max_length is not None else self.max_len) - len(sequence)

                if self.padding_side == "right":
                    if return_attention_mask:
                        att_mask = [1] * len(sequence) + [0] * difference
                    if return_token_type_ids:
                        token_type_ids = (
                            token_type_ids + [self.pad_token_type_id] * difference
                        )
                    if return_special_tokens_mask:
                        token_mask = token_mask + [1] * difference
                    sequence = sequence + [self.pad_token_id] * difference
                
                elif self.padding_side == "left":
                    if return_attention_mask:
                        att_mask = [0] * difference + [1] * len(sequence)
                    if return_token_type_ids:
                        token_type_ids = [self.pad_token_type_id] * difference + token_type_ids
                    if return_special_tokens_mask:
                        token_mask = [1] * difference + token_mask
                    sequence = [self.pad_token_id] * difference + sequence
                
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))
            elif return_attention_mask:
                att_mask = [1] * len(sequence)
            
            seq_list.append(sequence)
            if return_token_type_ids:
                type_list.append(token_type_ids)
            if return_special_tokens_mask:
                token_list.append(token_mask)
            if return_attention_mask:    
                attention_list.append(att_mask)

        encoded_inputs["input_ids"]=seq_list
        if return_special_tokens_mask: encoded_inputs["special_tokens_mask"] = token_list
        if return_token_type_ids: encoded_inputs["token_type_ids"] = type_list
        if return_attention_mask: encoded_inputs["attention_mask"] = attention_list
        return encoded_inputs
        
    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        max_length=None,
        add_special_tokens=True,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
    ):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens and manages a window stride for
        overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful for edge effect removal when using sequential
                list of inputs.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - False: Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default True).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default True)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:
                ``input_ids``: list of token ids to be fed to a model
                ``token_type_ids``: list of token type ids to be fed to a model

                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        # Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if max_length == None:  max_length = self.max_len 
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Handle special_tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)

        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids

        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]

        if max_length is None and len(encoded_inputs["input_ids"]) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.max_len)
            )

        needs_to_be_padded = pad_to_max_length and (
            max_length
            and len(encoded_inputs["input_ids"]) < max_length
            or max_length is None
            and len(encoded_inputs["input_ids"]) < self.max_len
            and self.max_len <= 10000
        )

        if pad_to_max_length and max_length is None and self.max_len > 10000:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.max_len) - len(encoded_inputs["input_ids"])

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]

            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        elif return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        # Prepare inputs as tensors if asked
        if return_tensors == "tf" and is_tf_available():
            encoded_inputs["input_ids"] = tf.constant([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = tf.constant([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = tf.constant([encoded_inputs["attention_mask"]])

        elif return_tensors == "pt" and is_torch_available():
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return encoded_inputs

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs=None,
        add_special_tokens=False,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        return_tensors=None,
        return_input_lengths=False,
        return_attention_masks=False,
        return_offsets_mapping=False,
        context_split=False,
        **kwargs
    ):
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )
        
        batch_outputs = {}
        if context_split == True: 
            for ids_or_pair_ids in batch_text_or_text_pairs:
                if isinstance(ids_or_pair_ids, (list, tuple)):
                    assert len(ids_or_pair_ids) == 2
                    ids, pair_ids = ids_or_pair_ids
                else:
                    ids, pair_ids = ids_or_pair_ids, None
                outputs = self.encode_plus(
                    ids,
                    pair_ids,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                    stride=stride,
                    truncation_strategy=truncation_strategy,
                    return_tensors=None,
                    context_split=True
                )
                # Append the non-padded length to the output
                if return_input_lengths:
                    outputs["input_len"] = len(outputs["input_ids"])

                for key, value in outputs.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = []
                    batch_outputs[key].extend(value)
            
        else:
            for ids_or_pair_ids in batch_text_or_text_pairs:
                if isinstance(ids_or_pair_ids, (list, tuple)):
                    assert len(ids_or_pair_ids) == 2
                    ids, pair_ids = ids_or_pair_ids
                else:
                    ids, pair_ids = ids_or_pair_ids, None
                outputs = self.encode_plus(
                    ids,
                    pair_ids,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                    stride=stride,
                    truncation_strategy=truncation_strategy,
                    return_tensors=None,
                )

                # Append the non-padded length to the output
                if return_input_lengths:
                    outputs["input_len"] = len(outputs["input_ids"])

                for key, value in outputs.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = []
                    batch_outputs[key].append(value)

        # Compute longest sequence size
        max_seq_len = max(map(len, batch_outputs["input_ids"]))
        if return_attention_masks:
            # Allow the model to not give any special attention to padded input
            batch_outputs["attention_mask"] = [[0] * len(v) for v in batch_outputs["input_ids"]]

        if return_tensors is not None:

            # Do the tensor conversion in batch
            for key, value in batch_outputs.items():

                padded_value = value
                # verify that the tokenizer has a pad_token_id
                if key != "input_len" and self._pad_token is not None:
                    # Padding handle
                    padded_value = [
                        v + [self.pad_token_id if key == "input_ids" else 1] * (max_seq_len - len(v))
                        for v in padded_value
                    ]

                if return_tensors == "tf" and is_tf_available():
                    batch_outputs[key] = tf.constant(padded_value)
                elif return_tensors == "pt" and is_torch_available():
                    batch_outputs[key] = torch.tensor(padded_value)
                elif return_tensors is not None:
                    logger.warning(
                        "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                            return_tensors
                        )
                    )

        # encoder_attention_mask requires 1 for real token, 0 for padding, just invert value
        if return_attention_masks:
            if is_torch_available():
                batch_outputs["attention_mask"] = torch.abs(batch_outputs["attention_mask"] - 1)
            else:
                batch_outputs["attention_mask"] = tf.abs(batch_outputs["attention_mask"] - 1)
                
        return batch_outputs
    
    def save_vocabulary(self, vocab_path):

        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None): 
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1= None, already_has_special_tokens= False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def truncate_sequences(
        self, ids, pair_ids=None, num_tokens_to_remove=0, truncation_strategy="longest_first", stride=0
    ):
        """Truncates a sequence pair in place to the maximum length.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - False: Does not truncate (raise an error if the input sequence is longer than max_length)
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == "longest_first":
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == "only_first":
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == "only_second":
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == False:
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', False]"
            )
        return (ids, pair_ids, overflowing_tokens)
    
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return len(cls + token_ids_0 + sep) * [0]
            else:
                num_pieces = int(len(token_ids_0)//510) + 1
                return (len(cls + token_ids_0 + sep) + 2*(num_pieces-1)) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers). Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    def _clean_text(self, text):
            """Performs invalid character removal and whitespace cleanup on text."""
            output = []
            for char in text:
                cp = ord(char)
                if cp == 0 or cp == 0xFFFD or self._is_control(char):
                    continue
                if self._is_whitespace(char):
                    output.append(" ")
                else:
                    output.append(char)
            return "".join(output)
        
    def _is_whitespace(self,char):
            """Checks whether `chars` is a whitespace character."""
            # \t, \n, and \r are technically contorl characters but we treat them
            # as whitespace since they are generally considered as such.
            if char == " " or char == "\t" or char == "\n" or char == "\r":
                return True
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
            return False

    def _is_control(self,char):
            """Checks whether `chars` is a control character."""
            # These are technically control characters but we count them as whitespace
            # characters.
            if char == "\t" or char == "\n" or char == "\r":
                return False
            cat = unicodedata.category(char)
            if cat.startswith("C"):
                return True
            return False
        