# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
import pypinyin
from argparse import Namespace
from distutils.util import strtobool

import numpy
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect, ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer

from datamodule.transforms import TextTransform

class E2E(torch.nn.Module):
    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, odim, args, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if (
            self.rel_pos_type is None
            and args.transformer_encoder_attn_layer_type == "rel_mha"
        ):
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )

        idim = 80

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
        )

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        self.proj_decoder = None
        
        # 这里配置中相等，应该不会执行
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        # Transformer Decoder
        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")

        # self.lsm_weight = 0的时候会退化为CE Loss，这里的超参数默认设置为0.1
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        self.py_ctc_weight = args.py_ctc_weight
        
        # CTC
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None
            
        # 加载分词表
        hanzi_vocab_path = '/work/lixiaolou/program/baseline/spm/unigram/unigram5000_units.txt'  # 汉字分词表文件路径
        pinyin_vocab_path = '/work/lixiaolou/program/baseline/spm/unigram/pinyin.txt'  # 拼音分词表文件路径
        self.hanzi_vocab = load_vocab(hanzi_vocab_path)
        pinyin_vocab = load_vocab(pinyin_vocab_path)
        self.pinyin_vocab = {id_: pinyin for pinyin, id_ in pinyin_vocab.items()}
        # 拼音ctc
        py_text_transform = TextTransform(dict_path='/work/lixiaolou/program/baseline/spm/unigram/pinyin.txt')
        
        if self.py_ctc_weight > 0.0:
            self.py_ctc = CTC(
                len(py_text_transform.token_list), args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
            self.py_decoder = Decoder(
                odim=odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
            
        else:
            self.py_ctc = None

    def scorers(self):
        # 在predict的时候调用这个方法来计算分数
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, label):
        if self.transformer_input_layer == "conv1d":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        x, _ = self.encoder(x, padding_mask)

        # ctc loss
        loss_ctc, ys_hat = self.ctc(x, lengths, label)
        # ctc pinyin loss
        py_label = tensor_to_pinyin_ids(label, self.hanzi_vocab, self.pinyin_vocab)
        logging.debug(f'py_label: {py_label}')
        logging.debug(f'length: {lengths}')
        loss_py_ctc, ys_hat_pinyin = self.py_ctc(x, lengths, py_label)

        # 这里不会执行
        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        
        # pinyin decoder loss
        ys_pinyin_in_pad, ys_pinyin_out_pad = add_sos_eos(py_label, self.sos, self.eos, self.ignore_id)
        ys_pinyin_mask = target_mask(ys_pinyin_in_pad, self.ignore_id)
        pred_py_pad, _ = self.py_decoder(ys_pinyin_in_pad, ys_pinyin_mask, x, padding_mask)
        loss_py_att = self.criterion(pred_py_pad, ys_pinyin_out_pad)
        
        # rnnt loss
        
        # 总loss
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att + 0.2 * (self.py_ctc_weight * loss_py_ctc + (1 - self.py_ctc_weight) * loss_py_att)

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_py_ctc, loss_att, loss_py_att, acc


# 加载分词表
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token, idx = line.strip().split()
            vocab[int(idx)] = token
    return vocab

# 将ID转换为汉字
def ids_to_sentence(ids, vocab):
    hanzi = [vocab.get(int(id_), '<unk>') for id_ in ids]
    sentence = ''.join(hanzi)[1:]
    return sentence

# 将汉字转换为拼音ID
def hanzi_to_pinyin_ids(sentence, pinyin_vocab):
    pinyin_sentence = pypinyin.lazy_pinyin(sentence, style=pypinyin.Style.NORMAL)
    for py in pinyin_sentence:
        if py not in pinyin_vocab:
            logging.warning(f'py: {py}')
    return [pinyin_vocab.get(py, pinyin_vocab.get('<unk>')) for py in pinyin_sentence]

# 将张量转换为拼音ID张量
def tensor_to_pinyin_ids(tensor, hanzi_vocab, pinyin_vocab):
    max_length = 0
    id = []
    for i in range(tensor.size(0)):
        for j in range(tensor.size(1)):
            sentence_ids = tensor[i, j].tolist()
            # 忽略-1填充的部分
            sentence_ids = [id_ for id_ in sentence_ids if id_ != -1]
            sentence = ids_to_sentence(sentence_ids, hanzi_vocab)
            pinyin_ids = hanzi_to_pinyin_ids(sentence, pinyin_vocab)
            pinyin_ids.insert(0, 2)
            if max_length < len(pinyin_ids):
                max_length = len(pinyin_ids)
            id.append(pinyin_ids)
            # pinyin_ids_tensor[i, j, :len(pinyin_ids)] = torch.tensor(pinyin_ids, device=tensor.device)
    pinyin_ids_tensor = torch.full((tensor.size(0), tensor.size(1), max_length), fill_value=-1, device=tensor.device)
    for index, item in enumerate(id):
        pinyin_ids_tensor[index, 0, :len(item)] = torch.tensor(item, device=tensor.device)
    return pinyin_ids_tensor