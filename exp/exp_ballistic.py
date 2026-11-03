"""
Experiment class for ballistic trajectory prediction.
TSLib-style: supports all models under models/ directory.
"""
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from data_provider.data_factory import data_provider
from models import build_model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.provenance import masked_trajectory_metrics, write_evaluation_artifacts

warnings.filterwarnings('ignore')


class PhysicalCompositeLoss(nn.Module):
    def __init__(self, scale, ade_weight=0.5, fde_weight=0.25):
        super().__init__()
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).reshape(1, 1, -1)
        self.register_buffer('scale', scale_tensor)
        self.ade_weight = float(ade_weight)
        self.fde_weight = float(fde_weight)

    def forward(self, prediction, target, mask=None):
        error_m = (prediction - target) * self.scale
        if mask is None:
            mask = torch.ones(error_m.shape[:2], device=error_m.device)
        valid = mask.sum().clamp_min(1.0)
        rmse = torch.sqrt(
            (error_m.square().sum(-1) * mask).sum() / (valid * error_m.shape[-1]) + 1e-12
        )
        distance = torch.linalg.vector_norm(error_m, dim=-1)
        ade = (distance * mask).sum() / valid
        # Final displacement at the last valid step (the impact point when masked).
        last = (mask.sum(dim=1).long() - 1).clamp_min(0)
        fde = distance.gather(1, last[:, None]).mean()
        return rmse + self.ade_weight * ade + self.fde_weight * fde


class MaskedMSELoss(nn.Module):
    def forward(self, prediction, target, mask=None):
        squared = (prediction - target).square()
        if mask is None:
            return squared.mean()
        weights = mask[..., None].expand_as(squared)
        return (squared * weights).sum() / weights.sum().clamp_min(1.0)


class Exp_Ballistic:
    def __init__(self, args):
        self.args = args
        self.target_scale = np.ones(
            int(getattr(args, 'c_out', 3)),
            dtype=np.float32,
        )
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if torch.cuda.is_available() and getattr(self.args, 'use_gpu', 1):
            gpu_id = getattr(self.args, 'gpu', 0)
            print(f'Use GPU: cuda:{gpu_id}')
            return torch.device(f'cuda:{gpu_id}')
        print('Use CPU')
        return torch.device('cpu')

    def _build_model(self):
        model = build_model(self.args).float()
        total = sum(p.numel() for p in model.parameters())
        print(f'{self.args.model} params: {total:,}')
        return model

    def _predict(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_anchor):
        if str(getattr(self.args, 'output_mode', '')).lower() == 'physics_anchor':
            context = None
            if isinstance(batch_anchor, dict):
                context = batch_anchor['context'].float().to(self.device)
                batch_anchor = batch_anchor['anchor']
            anchor = batch_anchor.float().to(self.device)
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                              anchor=anchor, context=context)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if str(getattr(self.args, 'loss_mode', 'mse')).lower() == 'physical_composite':
            return PhysicalCompositeLoss(
                scale=self.target_scale,
                ade_weight=float(getattr(self.args, 'ade_loss_weight', 0.5)),
                fde_weight=float(getattr(self.args, 'fde_loss_weight', 0.25)),
            )
        return MaskedMSELoss()

    def _vali(self, loader, criterion):
        """Return (mean batch loss, global coordinate RMSE in meters) on a loader."""
        self.model.eval()
        total_loss = []
        scale = torch.as_tensor(self.target_scale, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        squared_error_sum = 0.0
        element_count = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_anchor in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_mask = batch_mask.float().to(self.device)

                # TSLib-style decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                outputs = self._predict(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_anchor)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y, batch_mask)
                total_loss.append(loss.item())
                error_m = (outputs - batch_y) * scale
                squared_error_sum += float(
                    (error_m.double().square().sum(-1) * batch_mask).sum().item()
                )
                element_count += int(batch_mask.sum().item()) * error_m.shape[-1]
        self.model.train()
        rmse_m = float(np.sqrt(squared_error_sum / max(element_count, 1)))
        return np.average(total_loss), rmse_m

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        self.target_scale = np.asarray(
            train_data.scaler.std,
            dtype=np.float32,
        )

        # Optional training subset
        if hasattr(self.args, 'train_subset') and self.args.train_subset > 0:
            n_sub = min(self.args.train_subset, len(train_data))
            train_loader = DataLoader(
                Subset(train_data, list(range(n_sub))),
                batch_size=self.args.batch_size, shuffle=True,
                num_workers=self.args.num_workers, pin_memory=True, drop_last=False)
            print(f"Training subset: {n_sub} samples")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        criterion = self._select_criterion().to(self.device)
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_anchor) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_mask = batch_mask.float().to(self.device)

                # TSLib-style decoder input: label_len known + zeros for pred_len
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                outputs = self._predict(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_anchor)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y, batch_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.0f}s")
            train_loss = np.average(train_loss)
            vali_loss, vali_rmse_m = self._vali(vali_loader, criterion)

            print(f"Epoch: {epoch+1} | Train: {train_loss:.6f} Vali: {vali_loss:.6f} "
                  f"Vali RMSE(m): {vali_rmse_m:.6f}")

            select_metric = str(getattr(self.args, 'select_metric', 'loss')).lower()
            selection_value = vali_rmse_m if select_metric == 'val_rmse_m' else vali_loss
            early_stopping(selection_value, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Load best model
        best_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        self.model.eval()
        preds, trues, masks = [], [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_anchor in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                outputs = self._predict(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_anchor)
                outputs = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, :].numpy()

                if test_data.scale:
                    outputs = test_data.inverse_transform(outputs.reshape(-1, 3)).reshape(outputs.shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, 3)).reshape(batch_y.shape)

                preds.append(outputs)
                trues.append(batch_y)
                masks.append(batch_mask.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks, axis=0)
        summary = masked_trajectory_metrics(preds, trues, masks)
        print(f"Test: RMSE={summary['rmse']:.4f} ADE={summary['ade_m']:.4f} FDE={summary['fde_m']:.4f}")

        write_evaluation_artifacts(
            args=self.args,
            setting=setting,
            preds=preds,
            trues=trues,
            test_evaluation_count=1,
            mask=None if masks.all() else masks,
        )
        return preds, trues
