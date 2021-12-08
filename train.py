import torch
from torch import nn
from torch.optim import Adam

from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from model.spnet import SPNet_1, SPNet_2, SPNet_3, SPNet_4, SPNet_5
from data_utils.spdataset import SPDataset
from data_utils.tracker import Tracker

import config

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def run_epoch(loaders, model, criterion, optimizer, train, prefix, tracker):
    if train:
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    else:
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        mse_tracker = tracker.track('{}_mse'.format(prefix), tracker_class(**tracker_params))
        r2_tracker = tracker.track('{}_r2'.format(prefix), tracker_class(**tracker_params))

    for fold_idx in range(len(loaders)):
        loader = loaders[fold_idx]
        for x, score_y in loader:
            x = x.to(device)
            score_y = score_y.to(device)

            if train:
                model.train()
                score_pred = model(x)
                optimizer.zero_grad()
                loss = criterion(score_pred, score_y)
                loss.backward()
                optimizer.step()
                loss_tracker.append(loss.item())
            else:
                model.eval()
                score_pred = model(x)
                mse = mean_squared_error(score_y.cpu().tolist(), score_pred.cpu().tolist())
                r2 = r2_score(score_y.cpu().tolist(), score_pred.cpu().tolist())
                mse_tracker.append(mse)
                r2_tracker.append(r2)

    if train:
        return loss_tracker.mean.value
    else:
        return {
            "r2": r2_tracker.mean.value,
            "mse": mse_tracker.mean.value
        }

def main():

    dataset = SPDataset(config.csv_path)
    folds = dataset.get_folds(k=10)

    for stage in range(len(folds)):
        
        model = models[config.n_layers-1](30).to(device)
        criterion = nn.MSELoss().to(device)
        optimizer = Adam(model.parameters(), lr=0.1)
        tracker = Tracker()

        best_scores = {
            "r2": float("-inf"),
            "mse": float("inf")
        }

        pbar = tqdm(range(config.epochs), desc='Stage {} '.format(stage+1), unit='it')

        for epoch in pbar:
            loss = run_epoch(folds[:-1], model, criterion, optimizer, True, "Training", tracker)
            scores = run_epoch([folds[-1]], model, criterion, optimizer, False, "Evaluating", tracker)

            pbar.set_postfix({"Loss": loss, "MSE": scores["mse"], "R2": scores["r2"]})
            if scores["mse"] < best_scores["mse"]:
                best_scores = scores
                torch.save({
                    "state_dict": model.state_dict(),
                    "scores": best_scores
                }, f"saved_models/best_spnet_{config.n_layers}_stage_{stage+1}.pth")

            pbar.set_postfix({"Loss": loss, "Best MSE": best_scores["mse"], "Best R2": best_scores["r2"]})
            pbar.update()

        print(f"Stage {stage+1} completed.")
        print("="*23)

        # swapping folds
        tmp = folds[-1]
        folds[1:] = folds[:-1]
        folds[0] = tmp

if __name__ == "__main__":
    models = [SPNet_1, SPNet_2, SPNet_3, SPNet_4, SPNet_5]
    main()