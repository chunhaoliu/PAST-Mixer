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
from utils.metrics import metric
from utils.provenance import write_evaluation_artifacts

warnings.filterwarnings('ignore')


class PhysicalCompositeLoss(nn.Module):
    def __init__(self, scale, ade_weight=0.5, fde_weight=0.25):
        super().__init__()
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).reshape(1, 1, -1)
        self.register_buffer('scale', scale_tensor)
        self.ade_weight = float(ade_weight)
        self.fde_weight = float(fde_weight)

    def forward(self, prediction, target):
        error_m = (prediction - target) * self.scale
        rmse = torch.sqrt(torch.mean(error_m.square()) + 1e-12)
        ade = torch.linalg.vector_norm(error_m, dim=-1).mean()
        fde = torch.linalg.vector_norm(error_m[:, -1, :], dim=-1).mean()
        return rmse + self.ade_weight * ade + self.fde_weight * fde


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
        return nn.MSELoss()

    def _vali(self, loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # TSLib-style decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

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

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # TSLib-style decoder input: label_len known + zeros for pred_len
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.0f}s")
            train_loss = np.average(train_loss)
            vali_loss = self._vali(vali_loader, criterion)

            print(f"Epoch: {epoch+1} | Train: {train_loss:.6f} Vali: {vali_loss:.6f}")

            early_stopping(vali_loss, self.model, path)
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
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, :].numpy()

                if test_data.scale:
                    outputs = test_data.inverse_transform(outputs.reshape(-1, 3)).reshape(outputs.shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, 3)).reshape(batch_y.shape)

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape = metric(preds, trues)
        print(f'Test: MSE={mse:.4f} MAE={mae:.4f} RMSE={rmse:.4f}')

        write_evaluation_artifacts(
            args=self.args,
            setting=setting,
            preds=preds,
            trues=trues,
            test_evaluation_count=1,
        )
        return preds, trues
