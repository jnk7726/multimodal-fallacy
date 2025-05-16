import lightning as L
import torch as th
from typing import Dict
from torchmetrics import MetricCollection
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
#changed



class MAMKitLightingModel(L.LightningModule):
    def __init__(
            self,
            model: th.nn.Module,
            loss_function,
            num_classes: int,
            optimizer_class,
            val_metrics: MetricCollection = None,
            test_metrics: MetricCollection = None,
            log_metrics: bool = True,
            **optimizer_kwargs
    ):
        super().__init__()

        self.model = model
        self.loss_function = loss_function()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes
        self.log_metrics = log_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        #changed Global class-level attention store
        self.attn_storage = defaultdict(list)

    def forward(
            self,
            x
    ):
        return self.model(x)

    def training_step(
            self,
            batch,
            batch_idx
    ):
        inputs, y_true = batch
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)

        self.log(name='train_loss', value=loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
            self,
            batch,
            batch_idx
    ):
        inputs, y_true = batch
        
        #changed
        # model = self.model
        
        # # Only visualize for MulTA model, and only for batch 0
        # if batch_idx == 0 and hasattr(model, 'text_crossmodal_blocks'):
        #     try:
        #         text_feat = model.positional_encoder(inputs['text_inputs'])
        #         audio_feat = model.positional_encoder(inputs['audio_inputs'])
        #         audio_mask = inputs['audio_input_mask']
    
        #         block = model.text_crossmodal_blocks[0]
        #         _, attn_weights = block(text_feat, audio_feat, audio_mask, return_attn=True)
        #         avg_attn = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
    
        #         import matplotlib.pyplot as plt
        #         import seaborn as sns
        #         import os
    
        #         # Save in the experiment directory
        #         save_dir = Path("/scratch/jnk7726/bdml/project_files/mamkit/demos/attention_maps")
        #         out_path = save_dir / f"attn_heatmap_epoch{self.current_epoch}.png"
    
        #         plt.figure(figsize=(10, 6))
        #         sns.heatmap(avg_attn, cmap="viridis")
        #         plt.title("Text → Audio Cross-Attention Heatmap")
        #         plt.xlabel("Audio Tokens")
        #         plt.ylabel("Text Tokens")
        #         plt.tight_layout()
        #         plt.savefig(out_path)
        #         plt.close()
        #     except Exception as e:
        #         self.print(f"[Visualization Skipped] {e}")
        
        # changed 2
        
        # Run model to get predictions and optionally attention
        # model = self.model
        # if hasattr(model, 'text_crossmodal_blocks') and batch_idx < 10:
        #     try:
        #         text_feat = model.positional_encoder(inputs['text_inputs'])
        #         audio_feat = model.positional_encoder(inputs['audio_inputs'])
        #         audio_mask = inputs['audio_input_mask']
        
        #         block = model.text_crossmodal_blocks[0]
        #         _, attn_weights = block(text_feat, audio_feat, audio_mask, return_attn=True)
        
        #         # Ensure attention shape is (B, T_text, T_audio)
        #         if attn_weights.ndim == 4:
        #             avg_attn = attn_weights.mean(dim=1)  # average over heads
        #         elif attn_weights.ndim == 3:
        #             avg_attn = attn_weights
        #         else:
        #             self.print(f"[Attention Skipped] Unexpected attention shape: {attn_weights.shape}")
        #             raise ValueError("Invalid attention shape")
        
        #         # Run prediction
        #         logits = model(inputs)
        #         preds = th.argmax(logits, dim=-1).cpu().numpy()
        #         true_labels = y_true.cpu().numpy()
        
        #         for i in range(len(preds)):
        #             attn_map = avg_attn[i].detach().cpu().numpy()
        #             pred_label = preds[i]
        #             true_label = true_labels[i]
        
        #             # Fix for accidental singleton dims
        #             if attn_map.ndim == 3:
        #                 self.print(f"[Attention Fix] Map was 3D for idx {i}, squeezing")
        #                 attn_map = attn_map.squeeze(0)
        #             if attn_map.ndim == 1:
        #                 self.print(f"[Attention Skipped] Map is 1D (flattened vector), idx {i}")
        #                 continue
        #             if attn_map.ndim != 2:
        #                 self.print(f"[Attention Skipped] Unexpected shape {attn_map.shape} at idx {i}")
        #                 continue
        
        #             # Save heatmap image
        #             save_dir = Path("/scratch/jnk7726/bdml/project_files/mamkit/demos/attention_maps")
        #             save_dir.mkdir(parents=True, exist_ok=True)
        #             img_path = save_dir / f"epoch{self.current_epoch}_b{batch_idx}_i{i}_pred{pred_label}_true{true_label}.png"
        
        #             plt.figure(figsize=(10, 6))
        #             sns.heatmap(attn_map, cmap="viridis")
        #             plt.title(f"Epoch {self.current_epoch} - Pred: {pred_label}, True: {true_label}")
        #             plt.xlabel("Audio Tokens")
        #             plt.ylabel("Text Tokens")
        #             plt.tight_layout()
        #             plt.savefig(img_path)
        #             plt.close()
        
        #             # Accumulate for class-wise averaging
        #             self.attn_storage[pred_label].append(attn_map)
        
        #     except Exception as e:
        #         self.print(f"[Attention Skipped] {e}")
        
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)

        self.log(name='val_loss', value=loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.val_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_metrics.update(y_hat, y_true)

        return loss

    def on_validation_epoch_end(
            self
    ) -> None:
        if self.val_metrics is not None:
            metric_values = self.val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'val_{key}', value, prog_bar=self.log_metrics)
            self.val_metrics.reset()
            
        #changed Save averaged attention maps per predicted class (if collected)
        # Save average attention per predicted class
        # save_dir = Path("/scratch/jnk7726/bdml/project_files/mamkit/demos/attention_maps")
        # save_dir.mkdir(parents=True, exist_ok=True)
        
        # for cls, maps in self.attn_storage.items():
        #     try:
        #         if not maps:
        #             continue
        
        #         # Group maps by their shape
        #         shape_to_maps = {}
        #         for m in maps:
        #             shape_to_maps.setdefault(m.shape, []).append(m)
        
        #         for i, (shape, same_shape_maps) in enumerate(shape_to_maps.items()):
        
        #             avg_map = np.mean(same_shape_maps, axis=0)
        #             path = save_dir / f"avg_attn_class{cls}_shape{i}_epoch{self.current_epoch}.png"
        
        #             plt.figure(figsize=(10, 6))
        #             sns.heatmap(avg_map, cmap="magma")
        #             plt.title(f"Avg Attention - Class {cls} - Shape {shape} - Epoch {self.current_epoch}")
        #             plt.xlabel("Audio Tokens")
        #             plt.ylabel("Text Tokens")
        #             plt.tight_layout()
        #             plt.savefig(path)
        #             plt.close()
        
        #     except Exception as e:
        #         self.print(f"[Avg Attention Save Failed] class={cls}: {e}")
        
        # self.attn_storage.clear()

    def test_step(
            self,
            batch,
            batch_idx
    ):
        # compute accuracy
        inputs, y_true = batch
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # new add
        # metrics = {}

        if self.test_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_metrics.update(y_hat, y_true)
            # new add
            # metrics.update(self.test_metrics.compute())
        # new add
        return loss
        # return {'test_loss': loss, **metrics}

    def on_test_epoch_end(
            self
    ) -> None:
        if self.test_metrics is not None:
            metric_values = self.test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'test_{key}', value, prog_bar=self.log_metrics)
            self.test_metrics.reset()

    def configure_optimizers(
            self
    ):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

