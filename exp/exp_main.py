import layers.graphs
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, LogSparseTransformer, FFTransformer, \
    LSTM, MLP, persistence, GraphTransformer, GraphLSTM, GraphFFTransformer, \
    GraphInformer, GraphLogSparse, GraphMLP, GraphAutoformer, GraphPersistence
from utils.tools import EarlyStopping, adjust_learning_rate, visual, PlotLossesSame
from utils.metrics import metric
from utils.graph_utils import data_dicts_to_graphs_tuple, split_torch_graph
from utils.CustomDataParallel import DataParallelGraph
from layers.Functionality import MultiQuantileLoss, temporal_split

import pickle
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.data == 'WindGraph':
            self.args.seq_len = self.args.label_len
        if self.args.quantiles is not None:
            self.args.quantiles = [float(q) for q in self.args.quantiles.split(',')]
            self.args.quantiles.sort() # acecending order
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'LogSparse': LogSparseTransformer,
            'FFTransformer': FFTransformer,
            'LSTM': LSTM,
            'MLP': MLP,
            'persistence': persistence,
            'GraphTransformer': GraphTransformer,
            'GraphLSTM': GraphLSTM,
            'GraphFFTransformer': GraphFFTransformer,
            'GraphInformer': GraphInformer,
            'GraphLogSparse': GraphLogSparse,
            'GraphMLP': GraphMLP,
            'GraphAutoformer': GraphAutoformer,
            'GraphPersistence': GraphPersistence,
        }

        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            if self.args.data == 'WindGraph':
                model = DataParallelGraph(model, device_ids=self.args.device_ids)
            else:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'quantile':
            quantiles = [float(q) for q in self.args.quantiles.split(',')]
            for q in quantiles:
                assert 0 < q < 1, f"Quantile value must be between 0 and 1, got {q}"
            criterion = MultiQuantileLoss(quantiles)
        else:
            criterion = nn.MSELoss()
        return criterion
    
    def MAPE(self, pred, tar, eps=1e-07):
        loss = torch.mean(torch.abs(pred - tar) / (tar + eps))
        return loss
    
    def evaluate_conformal_coverage(self, data, num_trials=100, calib_size=1000, val_size=500):
        self.model.eval()
        coverages = []
        interval_widths = []
        
        # Pre-compute predictions for all data
        with torch.no_grad():
            all_predictions = []
            all_true_values = []
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data:
                # Prepare inputs and get predictions
                if self.args.data == 'WindGraph':
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y
                    
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)
                    
                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                
                predictions = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, -self.args.c_out:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, -self.args.c_out:].to(self.device)
                
                all_predictions.append(predictions.cpu().numpy())
                all_true_values.append(batch_y.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_true_values = np.concatenate(all_true_values, axis=0)
        
        for _ in range(num_trials):
            # Split data into calibration and validation
            calib_indices, val_indices = self.temporal_split(np.arange(len(all_predictions)), calib_size, val_size)
            
            # Calculate qhat using calibration set
            qhat = self.calculate_qhat(all_predictions[calib_indices], all_true_values[calib_indices])
            
            # Evaluate coverage on validation set
            coverage, interval_width = self.evaluate_coverage(all_predictions[val_indices], all_true_values[val_indices], qhat)
            coverages.append(coverage)
            interval_widths.append(interval_width)
        
        # Visualize distribution of coverage
        plt.figure(figsize=(12, 6))
        
        # Coverage histogram
        plt.subplot(1, 2, 1)
        plt.hist(coverages, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(coverages), color='r', linestyle='dashed', linewidth=2)
        plt.title(f'Distribution of Coverage over {num_trials} trials')
        plt.xlabel('Coverage')
        plt.ylabel('Frequency')
        
        # Add mean and std text
        mean_coverage = np.mean(coverages)
        std_coverage = np.std(coverages)
        plt.text(0.05, 0.95, f'Mean: {mean_coverage:.3f}\nStd: {std_coverage:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # Normal probability plot
        plt.subplot(1, 2, 2)
        sorted_data = np.sort(coverages)
        norm_quantiles = norm.ppf(np.arange(1, len(coverages) + 1) / (len(coverages) + 1))
        plt.scatter(norm_quantiles, sorted_data)
        plt.plot(norm_quantiles, norm_quantiles, color='r', linestyle='--')
        plt.title('Normal Probability Plot of Coverage')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        
        plt.tight_layout()
        plt.show()
        
        # Print additional metrics
        target_coverage = 1 - (self.args.quantiles[0] - self.args.quantiles[-1])
        bias = mean_coverage - target_coverage
        rmse = np.sqrt(np.mean((np.array(coverages) - target_coverage) ** 2))
        mean_width = np.mean(interval_widths)
        
        print(f"Target Coverage: {target_coverage:.3f}")
        print(f"Mean Coverage: {mean_coverage:.3f}")
        print(f"Coverage Std: {std_coverage:.3f}")
        print(f"Bias: {bias:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Mean Interval Width: {mean_width:.3f}")
        
        return mean_coverage, std_coverage


    def calculate_qhat(self, predictions, true_values):
        lower_pred = predictions[:, :, 0]
        upper_pred = predictions[:, :, -1]
        
        scores = np.maximum(true_values - upper_pred, lower_pred - true_values)
        n = len(scores)
        alpha = 1 - (self.args.quantiles[0] - self.args.quantiles[-1])
        qhat = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n)
        
        return qhat

    def evaluate_coverage(self, predictions, true_values, qhat):
        lower_pred = predictions[:, :, 0] - qhat
        upper_pred = predictions[:, :, -1] + qhat
        
        in_interval = (true_values >= lower_pred) & (true_values <= upper_pred)
        coverage = np.mean(in_interval)
        
        interval_width = np.mean(upper_pred - lower_pred)
        
        return coverage, interval_width
    
    def calibrate_quantiles(self, setting):
        self.model.eval()
        
        calib_data, calib_loader = self._get_data(flag='calib')
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(calib_loader):
                # Prepare inputs (similar to test method)
                if self.args.data == 'WindGraph':
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y
                    
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)
                    
                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])

                # Get model predictions
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.view(outputs.size(0), outputs.size(1), self.args.c_out, -1)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        
        # Assuming preds has shape (samples, pred_len, num_quantiles)
        lower_idx, upper_idx = 0, -1  # Adjust these based on your quantile indices
        
        cal_lower = preds[:, :, lower_idx]
        cal_upper = preds[:, :, upper_idx]
        
        # Compute calibration scores
        cal_scores = np.maximum(trues - cal_upper, cal_lower - trues)
        
        # Compute the conformity score
        n = len(cal_scores)
        alpha = 1 - (self.args.quantiles[0] - self.args.quantiles[-1])
        qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
        
        # Correct the quantiles
        # corrected_lower = cal_lower - qhat
        # corrected_upper = cal_upper + qhat
        
        return qhat
    
    def vali(self, setting, vali_data, vali_loader, criterion, epoch=0, plot_res=1, save_path=None):
        total_loss = []
        total_mse = []
        total_mape = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                # Graph Data (i.e. Spatio-temporal)
                if self.args.data == 'WindGraph':
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])

                if self.args.data == 'WindGraph' and self.args.use_multi_gpu:
                    batch_x, sub_bs_x, target_gpus = split_torch_graph(batch_x, self.args.devices.split(','))
                    dec_inp, sub_bs_y, _ = split_torch_graph(dec_inp, self.args.devices.split(','))
                    assert np.array([sum(sub_i) == len(sub_i) for sub_i in [sub_bs_x[j] == sub_bs_y[j] for j in range(len(sub_bs_x))]]).all()

                    batch_x_mark = [batch_x_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    batch_y_mark = [batch_y_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Reshape outputs for quantile regression if necessary
                if self.args.loss == 'quantile':
                    outputs = outputs.view(outputs.size(0), outputs.size(1), self.args.c_out, -1)
                    # Assuming median is the middle quantile
                    median_index = len(self.args.quantiles.split(',')) // 2
                    outputs = outputs[:, :, :, median_index]
                else:
                    outputs = outputs.squeeze(-1)  # Remove last dimension for point prediction

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')     # self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if 'sistence' in self.args.model:       # Check if the model is the persistence model
            criterion = self._select_criterion()
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion)
            test_loss = self.vali(setting, test_data, test_loader, criterion)

            self.test('persistence_' + str(self.args.pred_len), test=0, base_dir='', save_dir='results/' + self.args.model + '/', save_flag=True)

            print('vali_loss: ', vali_loss)
            print('test_loss: ', test_loss)
            assert False

        self.vali_losses = []  # Store validation losses

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.checkpoint_flag:
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       checkpoint=self.args.checkpoint_flag, model_setup=self.args)

        if self.args.checkpoint_flag:
            load_path = os.path.normpath(os.path.join(path, 'checkpoint.pth'))
            if os.path.exists(load_path) and self.load_check(path=os.path.normpath(os.path.join(path, 'model_setup.pickle'))):
                self.model.load_state_dict(torch.load(load_path))
                epoch_info = pickle.load(
                    open(os.path.normpath(os.path.join('./checkpoints/' + setting, 'epoch_loss.pickle')), 'rb'))
                start_epoch = epoch_info['epoch']
                early_stopping.val_losses = epoch_info['val_losses']
                early_stopping.val_loss_min = epoch_info['val_loss_min']
                self.vali_losses = epoch_info['val_losses']
                del epoch_info
            else:
                start_epoch = 0
                print('Could not load best model')
        else:
            start_epoch = 0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        teacher_forcing_ratio = 0.8     # For LSTM Enc-Dec training (not used for others).
        total_num_iter = 0
        time_now = time.time()
        for epoch in range(start_epoch, self.args.train_epochs):

            # Reduce the tearcher forcing ration every epoch
            if self.args.model == 'LSTM':
                teacher_forcing_ratio -= 0.08
                teacher_forcing_ratio = max(0., teacher_forcing_ratio)
                print('teacher_forcing_ratio: ', teacher_forcing_ratio)

            # type4 lr scheduling is updated more frequently
            if self.args.lradj != 'type4':
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            train_loss = []

            self.model.train()
            epoch_time = time.time()
            num_iters = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.lradj == 'type4':
                    adjust_learning_rate(model_optim, total_num_iter + 1, self.args)
                    total_num_iter += 1
                if isinstance(batch_y, dict):
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    model_optim.zero_grad()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                else:
                    dec_inp = batch_y

                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                    dec_inp = dec_inp.float().to(self.device)

                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]

                # Note that the collate function is not optimised and might have some potential errors
                if self.args.data == 'WindGraph' and self.args.use_multi_gpu:
                    batch_x, sub_bs_x, target_gpus = split_torch_graph(batch_x, self.args.devices.split(','))
                    dec_inp, sub_bs_y, _ = split_torch_graph(dec_inp, self.args.devices.split(','))
                    assert np.array([sum(sub_i) == len(sub_i) for sub_i in [sub_bs_x[j] == sub_bs_y[j] for j in range(len(sub_bs_x))]]).all()

                    batch_x_mark = [batch_x_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    batch_y_mark = [batch_y_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    teacher_forcing_ratio = [teacher_forcing_ratio for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)
                
                # Reshape outputs for quantile regression if necessary
                if self.args.loss == 'quantile':
                    outputs = outputs.view(outputs.size(0), outputs.size(1), self.args.c_out, -1)
                else:
                    outputs = outputs.squeeze(-1)  # Remove last dimension for point prediction

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if isinstance(batch_y, layers.graphs.GraphsTuple):
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0 and self.args.verbose == 1:
                    print("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, num_iters, epoch + 1, np.average(train_loss)))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion, epoch=epoch, save_path=path)
            test_flag = False
            if test_flag:
                test_loss = self.vali(setting, test_data, test_loader, criterion, epoch=epoch, save_path=path)

            # Plot the losses:
            if self.args.plot_flag and self.args.checkpoint_flag:
                loss_save_dir = path + '/pic/train_loss.png'
                loss_save_dir_pkl = path + '/train_loss.pickle'
                if os.path.exists(loss_save_dir_pkl):
                    fig_progress = pickle.load(open(loss_save_dir_pkl, 'rb'))

                if 'fig_progress' not in locals():
                    fig_progress = PlotLossesSame(epoch + 1,
                                                  Training=train_loss,
                                                  Validation=vali_loss)
                else:
                    fig_progress.on_epoch_end(Training=train_loss,
                                              Validation=vali_loss)

                if not os.path.exists(os.path.dirname(loss_save_dir)):
                    os.makedirs(os.path.dirname(loss_save_dir))
                fig_progress.fig.savefig(loss_save_dir)
                pickle.dump(fig_progress, open(loss_save_dir_pkl, 'wb'))    # To load figure that we can append to

            if test_flag:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path, epoch)
            self.vali_losses += [vali_loss]       # Append validation loss
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # After Training, load the best model.
        if self.args.checkpoint_flag:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def load_check(self, path, ignore_vars=None, ignore_paths=False):
        # Function to check that the checkpointed and current settings are compatible.
        if ignore_vars is None:
            ignore_vars = [
                'is_training',
                'train_epochs',
                'plot_flag',
                'root_path',
                'data_path',
                'data_path',
                'checkpoints',
                'checkpoint_flag',
                'output_attention',
                'do_predict',
                'des',
                'n_closest',
                'verbose',
                'data_step',
                'itr',
                'patience',
                'des',
                'gpu',
                'use_gpu',
                'use_multi-gpu',
                'devices',
            ]
        if ignore_paths:
            ignore_vars += [
                'model_id',
                'test_dir',
            ]

        setting2 = pickle.load(open(path, 'rb'))
        for key, val in self.args.__dict__.items():
            if key in ignore_vars:
                continue
            if val != setting2[key]:
                print(val, ' is not equal to ', setting2[key], ' for ', key)
                return False

        return True

    def test(self, setting, test=1, base_dir='', save_dir=None, ignore_paths=False, save_flag=True):
        test_data, test_loader = self._get_data(flag='test')
        if save_dir is None:
            save_dir = base_dir
        if test:
            print('loading model')
            if len(base_dir) == 0:
                load_path = os.path.normpath(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            else:
                load_path = os.path.normpath(os.path.join(base_dir + 'checkpoints/' + setting, 'checkpoint.pth'))
            load_check_flag = self.load_check(path=os.path.normpath(os.path.join(os.path.dirname(load_path),
                                                                                 'model_setup.pickle')),
                                              ignore_paths=ignore_paths)
            if os.path.exists(load_path) and load_check_flag:
                self.model.load_state_dict(torch.load(load_path))
                # Extract quantile from setting string if present
                setting_parts = setting.split('_')
                if 'q' in setting_parts:
                    q_index = setting_parts.index('q')
                    if q_index + 1 < len(setting_parts):
                        self.args.quantile = float(setting_parts[q_index + 1])
            else:
                print('Could not load best model')

        preds = []
        trues = []
        station_ids = []
        if save_flag:
            if len(save_dir) == 0:
                folder_path = './test_results/' + setting + '/'
            else:
                folder_path = save_dir + 'test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.data == 'WindGraph':
                    if i == 1:
                        print('batch_x:', batch_x)
                        print('batch_y:', batch_y)
                        print('batch_x_mark:', batch_x_mark)
                        print('batch_y_mark:', batch_y_mark)
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                else:
                    dec_inp = batch_y

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)
                    dec_inp = dec_inp.float().to(self.device)

                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]

                if self.args.data == 'WindGraph' and self.args.use_multi_gpu:
                    batch_x, sub_bs_x, target_gpus = split_torch_graph(batch_x, self.args.devices.split(','))
                    dec_inp, sub_bs_y, _ = split_torch_graph(dec_inp, self.args.devices.split(','))
                    assert np.array([sum(sub_i) == len(sub_i) for sub_i in [sub_bs_x[j] == sub_bs_y[j] for j in range(len(sub_bs_x))]]).all()

                    batch_x_mark = [batch_x_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    batch_y_mark = [batch_y_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                if self.args.loss == 'quantile':
                    outputs = outputs.view(outputs.size(0), outputs.size(1), self.args.c_out, -1)
                else:
                    outputs = outputs.squeeze(-1)  # Remove last dimension for point prediction

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if self.args.data == 'WindGraph':
                    station_ids.append(batch_x.station_names)
                if i % 20 == 0:
                    if self.args.data == 'WindGraph':
                        input = batch_x.nodes.detach().cpu().numpy()
                    else:
                        input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if save_flag:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        if self.args.data == 'WindGraph':
            station_ids = np.concatenate(station_ids)

        print('test shape:', preds.shape, trues.shape)

        # save results
        if save_flag:
            folder_path = os.path.join(save_dir, 'results', setting) if save_dir else os.path.join('./results', setting)
            os.makedirs(folder_path, exist_ok=True)

        losses = {}

        if self.args.loss == 'quantile':
            quantiles = [float(q) for q in self.args.quantiles.split(',')]
            for i, q in enumerate(quantiles):
                mae, mse, rmse, mape, mspe = metric(preds[..., i], trues)
                losses.update({
                    f'mae_q{q}': mae,
                    f'mse_q{q}': mse,
                    f'rmse_q{q}': rmse,
                    f'mape_q{q}': mape,
                    f'mspe_q{q}': mspe,
                })
                
                # Inverse transform and calculate metrics for un-scaled data
                preds_un = test_data.inverse_transform(preds[..., i])
                trues_un = test_data.inverse_transform(trues)
                mae_un, mse_un, rmse_un, mape_un, mspe_un = metric(preds_un, trues_un)
                losses.update({
                    f'mae_un_q{q}': mae_un,
                    f'mse_un_q{q}': mse_un,
                    f'rmse_un_q{q}': rmse_un,
                    f'mape_un_q{q}': mape_un,
                    f'mspe_un_q{q}': mspe_un,
                })
                
                if save_flag:
                    np.save(os.path.join(folder_path, f'pred_q{q}.npy'), preds[..., i])
                    np.save(os.path.join(folder_path, f'pred_un_q{q}.npy'), preds_un)
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            preds_un = test_data.inverse_transform(preds)
            trues_un = test_data.inverse_transform(trues)
            mae_un, mse_un, rmse_un, mape_un, mspe_un = metric(preds_un, trues_un)
            
            losses = {
                'mae_sc': mae, 'mse_sc': mse, 'rmse_sc': rmse, 'mape_sc': mape, 'mspe_sc': mspe,
                'mae_un': mae_un, 'mse_un': mse_un, 'rmse_un': rmse_un, 'mape_un': mape_un, 'mspe_un': mspe_un,
            }
            
            if save_flag:
                np.save(os.path.join(folder_path, 'pred.npy'), preds)
                np.save(os.path.join(folder_path, 'pred_un.npy'), preds_un)

        if self.args.data == 'WindGraph':
            for stat in np.unique(station_ids):
                indxs_i = np.where(station_ids == stat)[0]
                if self.args.loss == 'quantile':
                    for i, q in enumerate(quantiles):
                        mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(preds[indxs_i, ..., i], trues[indxs_i])
                        preds_un_i = test_data.inverse_transform(preds[indxs_i, ..., i])
                        trues_un_i = test_data.inverse_transform(trues[indxs_i])
                        mae_un_i, mse_un_i, rmse_un_i, mape_un_i, mspe_un_i = metric(preds_un_i, trues_un_i)
                        losses.update({
                            f'mae_sc_{stat}_q{q}': mae_i, f'mse_sc_{stat}_q{q}': mse_i, 
                            f'rmse_sc_{stat}_q{q}': rmse_i, f'mape_sc_{stat}_q{q}': mape_i, 
                            f'mspe_sc_{stat}_q{q}': mspe_i,
                            f'mae_un_{stat}_q{q}': mae_un_i, f'mse_un_{stat}_q{q}': mse_un_i, 
                            f'rmse_un_{stat}_q{q}': rmse_un_i, f'mape_un_{stat}_q{q}': mape_un_i, 
                            f'mspe_un_{stat}_q{q}': mspe_un_i,
                        })
                else:
                    mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(preds[indxs_i], trues[indxs_i])
                    preds_un_i = test_data.inverse_transform(preds[indxs_i])
                    trues_un_i = test_data.inverse_transform(trues[indxs_i])
                    mae_un_i, mse_un_i, rmse_un_i, mape_un_i, mspe_un_i = metric(preds_un_i, trues_un_i)
                    losses.update({
                        f'mae_sc_{stat}': mae_i, f'mse_sc_{stat}': mse_i, f'rmse_sc_{stat}': rmse_i, 
                        f'mape_sc_{stat}': mape_i, f'mspe_sc_{stat}': mspe_i,
                        f'mae_un_{stat}': mae_un_i, f'mse_un_{stat}': mse_un_i, f'rmse_un_{stat}': rmse_un_i, 
                        f'mape_un_{stat}': mape_un_i, f'mspe_un_{stat}': mspe_un_i,
                    })

        if save_flag:
            np.save(os.path.join(folder_path, 'true.npy'), trues)
            np.save(os.path.join(folder_path, 'true_un.npy'), trues_un)
            if self.args.data == 'WindGraph':
                np.save(os.path.join(folder_path, 'station_ids.npy'), station_ids)
            
            with open(os.path.join(folder_path, "results_loss.txt"), 'w') as f:
                for key, value in losses.items():
                    f.write(f'{key}:{value}\n')
            
            with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
                if self.args.loss == 'quantile':
                    for q in quantiles:
                        f.write(f'Quantile {q}:\n')
                        for m in ['mse', 'mae', 'rmse', 'mape', 'mspe']:
                            f.write(f'{m}: {losses[f"{m}_q{q}"]}\n')
                        f.write('\n')
                else:
                    for m in ['mse', 'mae', 'rmse', 'mape', 'mspe']:
                        f.write(f'{m}: {losses[m + "_sc"]}\n')
        return losses