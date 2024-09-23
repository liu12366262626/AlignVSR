import re
import sys
sys.path.append('/work/liuzehua/task/VSR/cnvsrc')
from transformers import HubertModel
import torch
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
import torch
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.repeat import repeat
import torch.nn as nn
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from joblib import load


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)

# including 4 transformer layers
class ASR_frontend(torch.nn.Module):
    def __init__(self,
                 relu_type,
                 a_upsample_ratio,
                 attention_dim,
                 positional_dropout_rate,
                 args,
                 normalize_before=True,
                 concat_after=False,
                 positionwise_layer_type="linear",
                 ):
        super(ASR_frontend, self).__init__()
        
        pos_enc_class = RelPositionalEncoding


        self.pos_enc =  pos_enc_class(attention_dim, positional_dropout_rate)  # drop positional information randomly

        #encoder params
        attention_dim=args.adim
        attention_heads=args.aheads
        linear_units=args.eunits
        num_blocks = 4
        dropout_rate=args.dropout_rate
        attention_dropout_rate=args.transformer_attn_dropout_rate
        encoder_attn_layer_type=args.transformer_encoder_attn_layer_type
        macaron_style=args.macaron_style
        use_cnn_module=args.use_cnn_module
        zero_triu=getattr(args, "zero_triu", False)
        cnn_module_kernel=args.cnn_module_kernel

        # act pre_hook before load the model
        self._register_load_state_dict_pre_hook(_pre_hook)  

        # transformer part
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate) 



        if encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        #encoder is equal to multihead transformers
        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
            ),
        )


    def forward(self, x, masks):
        x = self.pos_enc(x)
        x, masks = self.encoders(x, masks)

        return x[0], masks


class ASR(torch.nn.Module):

    def __init__(self, odim, cfg, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        :param cfg is config
        :param current_epoch is current training epoch     
        """
        # init params
        # Check the relative positional encoding type
        args = cfg.model.visual_backbone
        self.cfg = cfg
        torch.nn.Module.__init__(self)
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.adim = args.adim
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        # init hubert(frozen) and k-means
        self.hubert_encoder = HubertModel.from_pretrained(cfg.hubert_model)
        self.kmeans = load(cfg.k_means_model)

        for param in self.hubert_encoder.parameters():
            param.requires_grad = False

        self.hubert_encoder = self.hubert_encoder.eval()
        self.compact_audio_memory = nn.Embedding(self.kmeans.n_clusters, 768)
        # init asr_frontend
        self.asr_frontend = ASR_frontend(relu_type = cfg.model.audio_backbone.relu_type, 
                                a_upsample_ratio = cfg.model.audio_backbone.a_upsample_ratio, 
                                attention_dim = cfg.model.audio_backbone.adim,
                                positional_dropout_rate = cfg.model.audio_backbone.dropout_rate,
                                args = cfg.model.audio_backbone,
                                )


        # init asr 6 transformer decoder and asr ctc
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

        self.ctc = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
        )



    
    def forward(self, audio, audio_length, label):
        audio = audio.squeeze(-1)
        B,T = audio.size()
        attention_mask = torch.arange(T, device = audio_length.device).expand(B, -1) < audio_length.unsqueeze(1)

        x = self.hubert_encoder(audio, attention_mask = attention_mask).last_hidden_state

        # Reshape x to (batch_size * time_steps, embedding_size)
        batch_size, time_steps, embedding_size = x.size()
        x_reshaped = x.view(batch_size * time_steps, embedding_size).detach().cpu()

        # Predict cluster center indices for each sample using the k-means model
        labels = self.kmeans.predict(x_reshaped)

        # Convert labels to a PyTorch tensor and ensure it is on the same device as the original tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=x.device).view(batch_size, time_steps)

        audio_feature = self.compact_audio_memory(labels_tensor)

        #audio mask
        audio_length = torch.div(audio_length, 322, rounding_mode="trunc")
        B,T,_ = audio_feature.size()    
        audio_padding_mask = torch.arange(T, device=audio.device).expand(B, -1) < audio_length.unsqueeze(1)
        audio_padding_mask = audio_padding_mask.unsqueeze(-2)
        audio_feature, _ = self.asr_frontend(audio_feature, audio_padding_mask)


        # ctc loss
        loss_ctc, ys_hat = self.ctc(audio_feature, audio_length, label)

        # add eos to ys_in_pad(model input) and eos to ys_out_pad(model output)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        # attention mask
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        decoder_feat, _ = self.decoder(ys_in_pad, ys_mask, audio_feature, audio_padding_mask)

        # calculate transformer loss
        loss_att = self.criterion(decoder_feat, ys_out_pad)
        # transformer decoder acc
        acc = th_accuracy(
            decoder_feat.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        att_w = self.cfg.loss.att_w
        ctc_w = self.cfg.loss.ctc_w

        loss = att_w * loss_att + ctc_w * loss_ctc 
        return loss, loss_att, loss_ctc, acc

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))
