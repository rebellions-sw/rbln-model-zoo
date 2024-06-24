import torch
import os
import typing
import numpy as np
import rebel

from .rbln_modeling_t5 import (
    T5ForConditionalGenerationRebelEncoder,
    T5ForConditionalGenerationRebelDecoder,
)

from transformers import T5Config
from transformers import BeamSearchScorer


def t5_compile_opt(
    input_seq: int,
    output_seq: int,
    weight_path: str,
    batch_size: typing.Optional[int] = 1,
    num_beams: typing.Optional[int] = 0,
):
    """
    Compile encoder and decoder models with cache enabled.

    Args:
        input_seq: T5-encoder sequence length
        output_seq: T5-decoder sequence length
        weight_path: huggingface T5 pretrained-model path
        batch_size: input batch size
        num_beams: T5-decoder beam size,
            if num_beams == 0, decoder batch size is 1
            else, decoder batch size is num_beams * batch_size

    Returns:
        compiled_model (RBLNCompiledModel):
            Compiled model that can be run on the RBLN ATOM
    """

    if batch_size != 1:
        raise NotImplementedError
    decoder_batch = batch_size if num_beams == 0 else batch_size * num_beams

    ### set configuration
    model_config = T5Config.from_pretrained(weight_path)
    model_config.rebel_enc_seq_len = input_seq
    model_config.rebel_dec_seq_len = output_seq
    model_config.torchscript = True
    model_config.rebel_opt_level = 1
    model_config.num_beams = num_beams

    n_layer = model_config.num_layers
    n_head = model_config.num_heads
    d_kv = model_config.d_kv

    model_encoder = T5ForConditionalGenerationRebelEncoder.from_pretrained(
        weight_path, config=model_config
    )
    model_decoder = T5ForConditionalGenerationRebelDecoder.from_pretrained(
        weight_path, config=model_config
    )

    ### generate scripted model
    # generate dummy encoder input
    enc_input_ids = torch.zeros((batch_size, input_seq), dtype=torch.long)
    enc_attn_mask = torch.ones((batch_size, input_seq), dtype=torch.long)

    enc_script_input = (enc_input_ids, enc_attn_mask)

    # generate dummy decoder input
    # cross-attn K, V output computed in encoder
    enc_cache_out = torch.zeros((n_layer, 2, 1, n_head, input_seq, d_kv), dtype=torch.float32)
    dec_input_ids = torch.zeros((decoder_batch, 1), dtype=torch.long)
    dec_attn_mask = torch.ones((1, output_seq), dtype=torch.long)
    dec_past_kv_tensor = torch.zeros(
        (decoder_batch, n_head, output_seq - 1, d_kv), dtype=torch.float32
    )

    dec_script_input = [enc_cache_out, enc_attn_mask, dec_input_ids, dec_attn_mask]
    dec_script_input += [
        dec_past_kv_tensor for i in range(n_layer * 2)
    ]  # extend to number of layer * 2(k,v)

    encoder_scripted_model = torch.jit.trace(model_encoder, enc_script_input).eval()
    decoder_scripted_model = torch.jit.trace(model_decoder, dec_script_input).eval()

    #### build rebel model
    # Create IR for each module
    encoder_ir = rebel.torchscript_to_ir(encoder_scripted_model, name="encoder")
    decoder_ir = rebel.torchscript_to_ir(decoder_scripted_model, name="decoder")
    decoder_ir.batch_size = decoder_batch

    # Caching encoder/decoder I/O
    connections = [(encoder_ir.outputs[0], decoder_ir.inputs[0])]
    for i in range(len(decoder_ir.outputs) - 1):
        connections.append((decoder_ir.outputs[i + 1], decoder_ir.inputs[i + 4]))

    # Compile
    compiled_model = rebel.compile(
        encoder_ir,
        decoder_ir,
        connections=connections,
    )

    return compiled_model


def t5_runtime_opt(rbln_fname: str):
    """
    Load RBLN models and return runnable object

    Args:
        rbln_encoder_fname: path of compiled model

    Returns:
        encoder_runtime (rebel.Runtime): runnable encoder object
        decoder_runtime (rebel.Runtime): runnable decoder object
    """
    compiled_model = rebel.RBLNCompiledModel(rbln_fname)
    encoder_runtime = compiled_model.create_runtime("encoder")
    decoder_runtime = compiled_model.create_runtime("decoder")

    return (encoder_runtime, decoder_runtime)


class T5BeamSearchOpt:
    """
    RBLN T5 decoding helper which supports greedy and beam search.
    Note that beam search scoring processed on host CPU

    Args:
        batch_size: input batch size
        num_beams: number of beams for beam search, if 0, greedy search will be conducted
        pad_token_id: id of token for pad
        eos_token_id: id of token for end of sentence
    """

    def __init__(self, batch_size: int, num_beams: int, pad_token_id: int, eos_token_id: int):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.decoder_batch_size = batch_size if num_beams == 0 else batch_size * num_beams

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        if num_beams == 0:
            self.beam_scorer = None
            self.beam_scores = None
        else:
            self.beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device="cpu",
                length_penalty=1.0,
                do_early_stopping=True,
                num_beam_hyps_to_keep=1,
            )
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float)
            beam_scores[:, 1:] = -1e9
            self.beam_scores = beam_scores.view((batch_size * num_beams,))

    def update_decoder_output(
        self, decoder_logit: np.ndarray, prev_output: torch.tensor, decoding_step: int
    ):
        """
        Post-process sinlge step and update output sequence

        Args:
            decoder_logit: logit of single step from T5-decoder output layer
                shape - [decoder_batch, 1, vocab_size],
            prev_output: stacked token sequence,
                shape - [decoder_batch, max_decoder_seq_length]
            decoding_step: current decoding step

        Returns:
            next_output (torch.tensor): updated token sequence
            beam_idx (List[int]): selected beam indice at current step
            end_decode (bool): Flags to determine stop decoding
                for greedy search, True if last token is 'eos_token_id'
                for beam search, True if all last tokens of beams are 'eos_token_id'
        """
        if self.num_beams == 0:
            beam_idx = []
            next_tokens = torch.argmax(torch.tensor(decoder_logit), dim=-1)

            next_output = prev_output
            next_output[:, decoding_step + 1] = next_tokens
            end_decode = False
        else:
            next_token_logits = torch.squeeze(torch.tensor(decoder_logit))
            next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            next_token_scores += self.beam_scores[:, None].expand_as(next_token_scores)
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            beam_outputs = self.beam_scorer.process(
                prev_output,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            self.beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            next_output = prev_output[beam_idx, :]
            next_output[:, decoding_step + 1] = beam_next_tokens.type(torch.long)

            end_decode = self.beam_scorer.is_done

        # break condition
        for i in range(self.decoder_batch_size):
            if next_output[i, decoding_step + 1] != self.eos_token_id:
                break
            end_decode = True
        return next_output, beam_idx, end_decode


class T5RBLNGeneration:
    """
    RBLN T5 inference helper,
    1) load module(s)
    2) pad input token sequence to make input of compiled encoder
    3) run encoder
    4) run decoder for iteration of maximum decoder sequence
        if possible, stop early (see T5BeamSearchOpt)
    Note that encoder & decoder compiled models with the target
    (enc_seq, dec_seq, num_beams) must exist.

    Args:
        rbln_model_path: directory of saved RBLNCompiled models
        enc_seq_list: list of encoder sequences
            input will be padded to nearest sequence in this list
            if input length is larger than max seq length in this list,
            input may be truncated.
        max_dec_seq_length: maximum number of decoding steps
        batch_size: input batch size
        num_beams: number of beams for beam search
            if 0, greedy search will be conducted
        start_token_id: id of initial decoder input token
        pad_token_id: id of token for pad
        eos_token_id: id of token for end of sentence
    """

    def __init__(
        self,
        rbln_model_path: str,
        enc_seq_list: typing.List[int],
        max_dec_seq_length: int,
        batch_size: typing.Optional[int] = 1,
        num_beams: typing.Optional[int] = 0,
        start_token_id: typing.Optional[int] = 0,
        pad_token_id: typing.Optional[int] = 0,
        eos_token_id: typing.Optional[int] = 3,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.dec_batch_size = batch_size if num_beams == 0 else batch_size * num_beams
        self.input_seq_list = enc_seq_list
        self.max_dec_seq_length = max_dec_seq_length
        self.enc_seq_list = sorted(enc_seq_list)
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # load module
        self.runtime_list = []
        for input_seq_length in enc_seq_list:
            rbln_compiled_path = os.path.join(
                rbln_model_path,
                f"i_seq_{input_seq_length}_o_seq_{max_dec_seq_length}_opt.rbln",
            )
            # load & connect enc-dec
            self.runtime_list.append(t5_runtime_opt(rbln_fname=rbln_compiled_path))

    def pad_to_enc_seq(self, input_ids: np.ndarray, input_attn_masks: np.ndarray):
        """
        Pad input and attention masks to proper length

        Args:
            input_ids: input token ids, shape - [batch, seq]
            input_attn_masks: input attention mask - [batch, seq]

        Returns:
            pad_input_ids (np.ndarray): padded input ids
            pad_attn_mask (np.ndarray): padded attention mask (valid = 1, pad = 0)
        """
        # input_ids : numpy array, [batch, seq]
        # input_attn_masks : numpy array, [batch, seq]
        source_length = input_ids.shape[-1]
        pad_length = self.enc_seq_list[-1]
        for i, seq_len in enumerate(self.enc_seq_list):
            if source_length < seq_len:
                pad_length = seq_len
                break

        if source_length <= pad_length:
            pad_input_ids = (
                np.ones([input_ids.shape[0], pad_length], dtype=np.long) * self.pad_token_id
            )
            pad_attn_mask = np.zeros([input_ids.shape[0], pad_length], dtype=np.long)
            pad_input_ids[:, :source_length] = input_ids
            pad_attn_mask[:, :source_length] = input_attn_masks
        else:
            pad_input_ids = input_ids[:, :pad_length]
            pad_attn_mask = input_attn_masks[:, :pad_length]
            pad_input_ids[:, -1] = self.eos_token_id

        return pad_input_ids, pad_attn_mask

    def generate(
        self,
        input_sample: typing.Union[list, np.ndarray],
        input_attn_masks: typing.Union[list, np.ndarray],
    ):
        """
        Run encoder and generate decoder sequence
        if num_beams == 0, decoder_batch_size = 1
        else, decoder_batch_size = batch_size * num_beams

        Args:
            input_sample: tokenized input sequence, shape - [batch_size, seq], type = long
            input_attn_masks: input attention mask, shape - [batch_size, seq], type = long

        Returns:
            dec_output (numpy.ndarray): generated decoder output sequence,
                shape - [decoder_batch_size, out_seq]
        """

        # pad input to candidate enc lengths
        enc_input_ids, enc_attn_mask = self.pad_to_enc_seq(
            np.asarray(input_sample), np.asarray(input_attn_masks)
        )

        runtime_id = self.enc_seq_list.index(enc_input_ids.shape[1])

        # set encoder and initial decoder inputs
        enc_input_ids = np.ascontiguousarray(enc_input_ids, dtype=np.long)
        enc_attn_mask = np.ascontiguousarray(enc_attn_mask, dtype=np.long)

        dec_attn_mask = torch.zeros((self.batch_size, self.max_dec_seq_length), dtype=torch.long)
        dec_output = (
            torch.ones(self.dec_batch_size, self.max_dec_seq_length + 1, dtype=torch.long)
            * self.start_token_id
        )

        num_beams = self.num_beams
        beam_idx = [0] * num_beams

        beam_search_wrapper = T5BeamSearchOpt(
            batch_size=self.batch_size,
            num_beams=self.num_beams,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        # select runtime based on encoder input sequence
        encoder_runtime, decoder_runtime = self.runtime_list[runtime_id]

        ## run encoder
        # encoder results are remained on device
        encoder_runtime.run(enc_input_ids, enc_attn_mask)

        ## run decoder
        for seq_idx in range(self.max_dec_seq_length):
            # update decoder input
            seq_beam_idx_arr = np.array(
                [seq_idx] + [beam_idx[i] for i in range(num_beams)] + [0] * (255 - num_beams)
            ).astype(np.int16)

            dec_input_ids = np.ascontiguousarray(
                dec_output[:, seq_idx : seq_idx + 1], dtype=np.long
            )
            dec_attn_mask[:, seq_idx] = 1
            dec_attn_mask = np.ascontiguousarray(dec_attn_mask, dtype=np.long)

            decoder_logit = decoder_runtime.run(
                enc_attn_mask, dec_input_ids, dec_attn_mask, seq_beam_idx_arr
            )

            beam_post_out = beam_search_wrapper.update_decoder_output(
                decoder_logit=decoder_logit,
                prev_output=dec_output,
                decoding_step=seq_idx,
            )
            dec_output, beam_idx, do_early_stop = beam_post_out

            if do_early_stop:
                dec_output = dec_output[:, : seq_idx + 1]
                break

        return dec_output.numpy()
