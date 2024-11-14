import sys
sys.path.append('./align_vsr')
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
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from align_vsr.Phase3_align_vsr.attention import (
    RelPositionMultiHeadedAttention,  # noqa: H301
    CrossMultiHeadedAttention,
    RelPositionCrossMultiHeadedAttention
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
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
import logging




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



# no smooth
class MaximizeAttentionLoss(nn.Module):
    def __init__(self, step, window_size):
        super(MaximizeAttentionLoss, self).__init__()
        self.step = step
        self.window_size = window_size

    def forward(self, attention_scores, target_indices, video_length):
        device = video_length.device

        total_loss = 0.0
        B,H,T_b,A = attention_scores[0].size()
        window_size = self.window_size
        step = self.step

        attention_score = torch.zeros_like(attention_scores[0], device = attention_scores[0].device)
        for item in attention_scores:    
            attention_score = attention_score + item
        attention_score = attention_score/len(attention_scores)  

        attention_score =torch.sum(attention_score, dim=1)/H


        attention_score = attention_score.reshape(-1, A)
        labels = torch.zeros_like(attention_score)

        for batch_idx in range(B):
            cam_label = target_indices[batch_idx]
            T = video_length[batch_idx]
            # 计算第一列的值
            # 生成从0到2*T的偶数（不包括2*T）
            centers = torch.arange(0, 2*T, step, device = device)
            # 使用整数类型的张量，并确保计算结果在整数类型中
            start_indices = torch.max(torch.zeros(T, dtype=torch.long, device = device), centers - (window_size // 2))
            start_indices = torch.min((2 * T - 1 - window_size) * torch.ones(T, dtype=torch.long, device = device), start_indices)
            indices = start_indices.unsqueeze(1) + torch.arange(window_size, dtype=torch.long, device = device)

            audio_indices = cam_label[indices]
            

            rows = torch.arange(start= T_b * batch_idx, end = T + T_b * batch_idx, device=device, dtype=torch.long).unsqueeze(1).expand(-1, window_size)


            for i in range(rows.shape[1]):  # 遍历列数
                row_col = rows[:, i]  # 获取 rows 的第 i 列
                audio_col = audio_indices[:, i]  # 获取 audio_indices 的第 i 列
                labels[row_col, audio_col] += 1  # 更新 labels 矩阵

        total_loss = -torch.sum(labels * torch.log(attention_score + 1e-8)) / torch.sum(video_length)


        return total_loss



class AudioBridgingModule(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate, num_layers):
        super(AudioBridgingModule, self).__init__()
        # 创建一个包含多个多头注意力层的列表
        self.cross_attention_layers = nn.ModuleList([
            RelPositionCrossMultiHeadedAttention(n_head=num_heads, n_feat=embed_size, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        #create pos_emb
        self.pos_enc_class = RelPositionalEncoding(embed_size, dropout_rate)




    def forward(self, visual_features, compact_audio_memory, mask):
        # 获取批次大小 B
        B = visual_features.size(0)
        
        _, pos_emb = self.pos_enc_class(compact_audio_memory.unsqueeze(0))
        pos_emb = pos_emb.to(visual_features.device)

    
        # 将 compact_audio_memory 移动到 visual_features 相同的设备
        compact_audio_memory = compact_audio_memory.to(visual_features.device)
        # 扩展 compact_audio_memory 到 (B, 200, D)
        expanded_audio_memory = compact_audio_memory.unsqueeze(0).expand(B, -1, -1)
        
        # 初始输入
        attention_input = visual_features
        
        # 初始的注意力权重列表
        attention_weights = []
        
        # 逐层应用多头注意力
        for layer in self.cross_attention_layers:
            attention_output, weights = layer(
                query=attention_input,
                key=expanded_audio_memory,
                value=expanded_audio_memory,
                mask=mask,  # 根据需要可以添加mask
                pos_emb = pos_emb,
                rtn_attn=True
            )
            # 更新下一层的输入为当前层的输出
            attention_input = attention_output
            # 保存当前层的注意力权重
            attention_weights.append(weights)
        
        # 返回最终的输出和所有层的注意力权重
        return attention_output, attention_weights
    
class Hybird_AVEncoder(nn.Module):
        def __init__(self, av_cross_encoder, video_encoder):
            super(Hybird_AVEncoder, self).__init__()
            self.av_cross_encoder = av_cross_encoder
            self.video_encoder = video_encoder


        def forward(self, xs, compact_audio_memory, mask):
            pos_emb = xs[1]
            xs, att_w = self.av_cross_encoder(xs[0], compact_audio_memory, mask)
            xs = (xs, pos_emb)
            xs, _ = self.video_encoder(xs, mask)

            return xs, att_w

# including VSR frontend + 4 layers Transformer + 4 layers hybird AVEncoder + 2 layers cross-attention
class VSR_frontend(torch.nn.Module):
    def __init__(self,args,cfg,
                normalize_before = True, 
                concat_after=False,) -> None:
        super(VSR_frontend, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)  
        # init params
        relu_type = getattr(args, "relu_type", "swish")
        attention_dim = attention_dim=args.adim
        positional_dropout_rate=args.dropout_rate
        linear_units=args.eunits
        dropout_rate=args.dropout_rate
        attention_heads=args.aheads
        attention_dropout_rate=args.transformer_attn_dropout_rate
        zero_triu=getattr(args, "zero_triu", False)
        cnn_module_kernel=args.cnn_module_kernel
        num_blocks = args.elayers
        use_cnn_module=args.use_cnn_module
        macaron_style=args.macaron_style


        # init vsr frontend
        pos_enc_class = RelPositionalEncoding
        self.frontend = Conv3dResNet(relu_type=relu_type)

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(512, attention_dim),
            pos_enc_class(attention_dim, positional_dropout_rate),  # drop positional information randomly 
        )
        
        # init vsr transformer part
        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate) 


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


        # 创建一个包含多个多头注意力层的列表
        self.Hybird_AVEncoder = Hybird_AVEncoder(
            av_cross_encoder = AudioBridgingModule(embed_size = cfg.model.Hybird_AVEncoder.abm.embed_size,
                                        num_heads = cfg.model.Hybird_AVEncoder.abm.num_heads,
                                        num_layers = cfg.model.Hybird_AVEncoder.abm.num_layers,
                                        dropout_rate = cfg.model.Hybird_AVEncoder.abm.dropout_rate),
            video_encoder = EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
            )
        )

        self.Hybird_AVEncoder_layers = nn.ModuleList([
            self.Hybird_AVEncoder
            for _ in range(cfg.model.Hybird_AVEncoder.num_layers)
        ])


        self.abm = AudioBridgingModule(embed_size = cfg.model.Hybird_AVEncoder.abm.embed_size,
                                        num_heads = cfg.model.Hybird_AVEncoder.abm.num_heads,
                                        num_layers = 2,
                                        dropout_rate = cfg.model.Hybird_AVEncoder.abm.dropout_rate)
        



    def forward(self, xs, masks, compact_audio_memory):

        xs = self.frontend(xs)

        xs = self.embed(xs)#a linear layer ＋ positional code = 2 output

        xs, masks = self.encoders(xs, masks)
        
        att_w = []

        # 逐层应用多头注意力
        for layer in self.Hybird_AVEncoder_layers:
            xs, att_weights= layer(xs, compact_audio_memory, masks)
            att_w.extend(att_weights)
        
        pos = xs[1]
        xs, att_weights = self.abm(xs[0], compact_audio_memory, masks)
        att_w.extend(att_weights)


        return xs, masks, pos, att_w




class V2A(torch.nn.Module):


    def __init__(self, odim, cfg, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        :param cfg is config
        :param current_epoch is current training epoch     
        """
        # init params
        args = cfg.model.visual_backbone
        self.cfg = cfg
        torch.nn.Module.__init__(self)
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



        # init vsr_frontend 
        self.vsr_frontend = VSR_frontend(args, cfg)

        # init compact audio memory and ABM 
        ckpt = torch.load(
                self.cfg.CAM_path, map_location=lambda storage, loc: storage
            )

        self.cam = ckpt['state_dict']['model.compact_audio_memory.weight']
        # Ensure the compact audio memory does not update during training
        self.cam.requires_grad = False

        

        # init vsr decoder and vsr ctc
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

        self.v2a_loss = MaximizeAttentionLoss(cfg.loss.a2v_attscore.step, cfg.loss.a2v_attscore.window_size)





    def forward(self, video, video_length, label, audio_label):

        # add eos to ys_in_pad(model input) and eos to ys_out_pad(model output)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        # attention mask
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        # video mask
        video_padding_mask = make_non_pad_mask(video_length).to(video.device).unsqueeze(-2)
        video_feature, _ , _, att_weights= self.vsr_frontend(video, video_padding_mask, self.cam)




        a2v_attscore_loss =self.v2a_loss(att_weights, audio_label, video_length)

        # ctc loss
        loss_ctc, ys_hat = self.ctc(video_feature, video_length, label)


        # asr2vsr att loss

        decoder_feat, _ = self.decoder(ys_in_pad, ys_mask, video_feature, video_padding_mask)

        # calculate transformer loss
        asr2vsr_loss_att = self.criterion(decoder_feat, ys_out_pad)
        # transformer decoder acc
        asr2vsr_acc = th_accuracy(
            decoder_feat.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )



        asr2vsr_att_w = self.cfg.loss.asr2vsr_att_w
        ctc_w = self.cfg.loss.ctc_w
        a2v_attscore_loss_w = self.cfg.loss.a2v_attscore.a2v_attscore_w

        loss = asr2vsr_att_w * asr2vsr_loss_att + ctc_w * loss_ctc + a2v_attscore_loss_w * a2v_attscore_loss 


        return loss, asr2vsr_loss_att, asr2vsr_acc, loss_ctc, a2v_attscore_loss
    
    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

