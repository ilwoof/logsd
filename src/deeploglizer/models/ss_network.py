import torch
import torch.nn.functional as F
from torch import nn

from src.deeploglizer.common.utils import new_mask_feature, mask_vectors_in_batch_by_duplicate_node
from src.deeploglizer.models import ForcastBasedModel
from src.deeploglizer.models.module import Encoder, Decoder


class SS_Net(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        embedding_dim=16,
        hidden_size=100,
        encoder_type='linear',
        decoder_type='linear',
        num_layers=1,            # just for LSTM
        num_directions=2,        # just for LSTM 
        kernel_sizes=[2, 3, 4],  # just for CNN
        pooling_mode='max',      # just for CNN
        reconstruction_mode='global2local',
        masking_by='frequency',
        masking_mode='no_mask',
        masking_ratio=0.5,
        loss_ablation='all',
        feature_transpose=False,
        model_save_path="./ae_models",
        feature_type="semantics",
        label_type="none",
        eval_type="session",
        topk=5,
        embedding_type='tfidf',
        freeze=False,
        gpu=-1,
        patience=3,
        batch_size=64,
        window_size=500,
        epoches=100,
        warmup_epoch=3,
        peak_lr=1e-2,
        end_lr=2e-4,
        weight_decay=1e-4,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            encoder_type=encoder_type,
            hidden_size=hidden_size,
            kernel_sizes=kernel_sizes,
            num_directions=num_directions,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            embedding_type=embedding_type,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            patience=patience,
            batch_size=batch_size,
            warmup_epoch=warmup_epoch,
            peak_lr=peak_lr,
            end_lr=end_lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.kernel_sizes = kernel_sizes
        self.pooling_mode = pooling_mode
        # self.act_function = nn.LeakyReLU(inplace=True, negative_slope=2.5e-1)  # nn.ReLU()
        self.reconstruction_mode = reconstruction_mode
        self.masking_by = masking_by
        self.masking_mode = masking_mode
        self.masking_ratio = masking_ratio
        self.feature_transpose = feature_transpose
        self.loss_ablation = loss_ablation
        self.window_size = window_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.encoder1 = Encoder(encoder_type=encoder_type,
                                decoder_type=decoder_type,
                                pooling_mode=pooling_mode,
                                feature_transpose=feature_transpose,
                                window_size=window_size,
                                input_dim=embedding_dim,
                                hidden_dim=hidden_size,
                                kernel_sizes=kernel_sizes,
                                num_layers=num_layers
                                )

        self.encoder2 = Encoder(encoder_type=encoder_type,
                                decoder_type=decoder_type,
                                pooling_mode=pooling_mode,
                                feature_transpose=feature_transpose,
                                window_size=window_size,
                                input_dim=embedding_dim,
                                hidden_dim=hidden_size,
                                kernel_sizes=kernel_sizes,
                                num_layers=num_layers
                                )

        self.decoder = Decoder(encoder_type=encoder_type,
                               decoder_type=decoder_type,
                               feature_transpose=feature_transpose,
                               window_size=window_size,
                               input_dim=hidden_size,
                               output_dim=embedding_dim,
                               kernel_sizes=kernel_sizes,
                               num_layers=num_layers
                               )

        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, input_dict):
        if self.encoder_type == 'cnn':
            x = input_dict["features"]
            if self.embedding_dim == 1:
                x = x.unsqueeze(-1)
            else:
                x = self.embedder(x).float()
            if self.masking_by == 'probability':
                masked_x, mask = new_mask_feature(x, mode='row', p=self.masking_ratio, fill_value=0.0)
            elif self.masking_by == 'frequency':
                masked_x, mask = mask_vectors_in_batch_by_duplicate_node(x, p=self.masking_ratio, fill_value=0.0)
            else:  # self.masking_by == 'no_masking':
                masked_x = x
                mask = torch.ones_like(x, dtype=torch.bool, device=x.device)

        if self.encoder_type == 'cnn':
            x = x.unsqueeze(1).repeat(1, len(self.kernel_sizes), 1, 1)
            mask = mask.unsqueeze(1).repeat(1, len(self.kernel_sizes), 1, 1)
            masked_x = masked_x.unsqueeze(1).repeat(1, len(self.kernel_sizes), 1, 1)

            if self.reconstruction_mode in ['global', 'global2local'] or self.masking_by == 'no_masking' or self.masking_ratio == 0.0:
                z, _, pooled_z = self.encoder1(x)
            else:
                z, _, pooled_z = self.encoder1(masked_x)
            masked_z, _, pooled_masked_z = self.encoder2(masked_x)
            x_recst = self.decoder(z)

            if self.reconstruction_mode in ['global'] or self.masking_by == 'no_masking' or self.masking_ratio == 0.0:
                rect_error = self.criterion(x, x_recst).mean(dim=-1).mean(dim=-1).mean(dim=-1)
            else:
                rect_error = self.criterion(x.masked_fill(mask, 0), x_recst.masked_fill(mask, 0)).mean(dim=-1).mean(dim=-1).mean(dim=-1)

            if self.loss_ablation in ['no_rect', 'projection_only']:
                pred = self.criterion(pooled_z, pooled_masked_z).mean(dim=-1)
            else:
                pred = self.criterion(pooled_z.detach(), pooled_masked_z).mean(dim=-1)

            oc_dist = self.criterion(pooled_z, self.center.detach()).mean(dim=-1)

        if self.loss_ablation == 'no_pj':
            loss = self.alpha * rect_error.mean() + self.gmma * oc_dist.mean()
        elif self.loss_ablation == 'no_oc':
            loss = self.alpha * rect_error.mean() + self.beta * pred.mean()
        elif self.loss_ablation == 'no_rect':
            loss = self.beta * pred.mean() + self.gmma * oc_dist.mean()
        elif self.loss_ablation == 'all':
            loss = self.alpha * rect_error.mean() + self.beta * pred.mean() + self.gamma * oc_dist.mean()
        elif self.loss_ablation == 'rect_only':
            loss = rect_error.mean()
        elif self.loss_ablation == 'oc_only':
            loss = oc_dist.mean()
        elif self.loss_ablation == 'projection_only':
            loss = pred.mean()
        else:
            RuntimeError("The specified loss is not supported!")

        return_dict = {"loss": loss, "y_pred": pred, "n_center": pooled_z, "r_loss": rect_error.mean()*1000, "p_loss": pred.mean()*100, "o_loss": oc_dist.mean()}

        return return_dict
