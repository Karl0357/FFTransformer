import os
import torch
import numpy as np
from exp.exp_main import Exp_Main
from models import GraphFFTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.graph_utils import data_dicts_to_graphs_tuple

def load_model(args, setting):
    exp = Exp_Main(args)
    exp.model = exp._build_model()
    load_path = os.path.normpath(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
    exp.model.load_state_dict(torch.load(load_path))
    exp.model.eval()
    return exp

def generate_predictions(exp, test_data, test_loader):
    preds = []
    exp.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = data_dicts_to_graphs_tuple(batch_x, device=exp.device)
            dec_inp = data_dicts_to_graphs_tuple(batch_y, device=exp.device)

            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_zeros = dec_inp.nodes[:, (exp.args.label_len - 1):exp.args.label_len, :]
            dec_zeros = dec_zeros.repeat(1, exp.args.pred_len, 1).float()
            dec_zeros = torch.cat([dec_inp.nodes[:, :exp.args.label_len, :], dec_zeros], dim=1)
            dec_zeros = dec_zeros.float().to(exp.device)
            dec_inp = dec_inp.replace(nodes=dec_zeros)

            dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -exp.args.dec_in:])
            batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -exp.args.enc_in:])

            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -exp.args.c_out
            outputs = outputs[:, -exp.args.pred_len:, f_dim:]
            preds.append(outputs.detach().cpu().numpy())

    preds = np.vstack(preds)
    return preds

def combine_predictions(args, settings):
    # Load test data
    _, test_loader = Exp_Main(args)._get_data(flag='test')

    # Load models and generate predictions
    lower_exp = load_model(args, settings['lower'])
    upper_exp = load_model(args, settings['upper'])
    point_exp = load_model(args, settings['point'])

    lower_preds = generate_predictions(lower_exp, test_data, test_loader)
    upper_preds = generate_predictions(upper_exp, test_data, test_loader)
    point_preds = generate_predictions(point_exp, test_data, test_loader)

    # Combine predictions
    combined_preds = np.stack([lower_preds, point_preds, upper_preds], axis=-1)

    # Save combined predictions
    np.save('./results/combined_predictions.npy', combined_preds)

    print(f"Combined predictions shape: {combined_preds.shape}")
    print("Combined predictions saved to ./results/combined_predictions.npy")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Combine predictions from multiple models')
    
    # Basic config
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='GraphFFTransformer', help='model name')
    parser.add_argument('--data', type=str, default='WindGraph', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset_example/DMI/dataset/', help='root path of the data file')
    parser.add_argument('--target', type=str, default='10KM_609_61', help='target variable in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # Quantile config
    parser.add_argument('--lower_quantile', type=float, default=0.05, help='lower quantile for prediction interval')
    parser.add_argument('--upper_quantile', type=float, default=0.95, help='upper quantile for prediction interval')

    # Model default parameters (you might need to add more based on your model's requirements)
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    # Modify settings to include quantile information
    settings = {
        'lower': f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_q{args.lower_quantile}_0',
        'upper': f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_q{args.upper_quantile}_0',
        'point': f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_0'
    }

    combine_predictions(args, settings)