import os
import argparse
from typing import List, Dict
from yacs.config import CfgNode
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from shapely.geometry import Point, Polygon

from utils import load_config
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from visualization.build_visualizer import Build_Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='config/TP/FlowChain/CIF_separate_cond_v_trajectron/zara2.yml',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="test")
    parser.add_argument(
        "--visualize", default=True, help="flag for whether visualize the results in mode:test")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()

W, H = 20, 10

from matplotlib.path import Path
import numpy as np

def compute_region_probability_from_samples_fast(sampled_seq, polygon_xy):
    """
    sampled_seq: (N_samples, T, 2)
    polygon_xy: list or array of (x, y)
    """
    N, T, _ = sampled_seq.shape

    poly_path = Path(polygon_xy)

    # reshape -> (N*T, 2)
    pts = sampled_seq.reshape(-1, 2)

    # bool mask, shape (N*T,)
    inside = poly_path.contains_points(pts)

    # reshape back -> (N, T)
    inside = inside.reshape(N, T)

    # 每个时刻的概率
    region_prob_seq = inside.mean(axis=0)

    return region_prob_seq


def train(cfg: CfgNode, save_model=True) -> None:
    validation = cfg.SOLVER.VALIDATION and cfg.DATA.TASK != "VP"

    data_loader = unified_loader(cfg, rand=True, split="train")
    if validation:
        val_data_loader = unified_loader(cfg, rand=False, split="val")
        val_loss = np.inf

    start_epoch = 0
    model = Build_Model(cfg)

    if model.check_saved_path():
        # model saved at the end of each epoch. resume training from next epoch
        start_epoch = model.load() + 1
        print('loaded pretrained model')

    if cfg.SOLVER.USE_SCHEDULER:
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=int(
                                                          cfg.SOLVER.ITER/10),
                                                      last_epoch=start_epoch-1,
                                                      gamma=0.7) for optimizer in model.optimizers]

    with tqdm(range(start_epoch, cfg.SOLVER.ITER)) as pbar:
        for i in pbar:
            loss_list = []
            for data_dict in data_loader:
                data_dict = {k: data_dict[k].cuda()
                             if isinstance(data_dict[k], torch.Tensor)
                             else data_dict[k]
                             for k in data_dict}

                loss_list.append(model.update(data_dict))

            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))

            # validation
            if (i+1) % cfg.SOLVER.SAVE_EVERY == 0:
                if validation:
                    curr_val_loss = evaluate_model(
                        cfg, model, val_data_loader)["score"]
                    if curr_val_loss < val_loss:
                        val_loss = curr_val_loss
                        if save_model:
                            model.save(epoch=i)
                else:
                    if save_model:
                        model.save(epoch=i)

        if cfg.SOLVER.USE_SCHEDULER:
            [scheduler.step() for scheduler in schedulers]
    return curr_val_loss


def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, visualize=False):
    model.eval()
    metrics = Build_Metrics(cfg)
    visualizer = Build_Visualizer(cfg)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    update_timesteps = [1]

    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})

    result_info = {}

    if visualize:
        with torch.no_grad():
            result_list = []
            print("timing the computation, evaluating probability map, and visualizing... ")
            data_loader_one_each = unified_loader(cfg, rand=False, split="test", batch_size=1)

            all_obs = []
            all_gt = []
            all_pred = []

            for i, data_dict in enumerate(tqdm(data_loader_one_each, leave=False, total=10)):
                data_dict = {k: data_dict[k].cuda()
                             if isinstance(data_dict[k], torch.Tensor)
                             else data_dict[k]
                             for k in data_dict}
                dict_list = []

                # result_dict = model.predict(deepcopy(data_dict), return_prob=True)  # warm-up
                torch.cuda.synchronize()
                starter.record()
                result_dict = model.predict(deepcopy(data_dict), return_prob=True)

                dict_list.append(deepcopy(result_dict))
                dict_list = metrics.denormalize(dict_list)  # denormalize the output

                # 收集轨迹
                obs_np = dict_list[0]["obs"].cpu().numpy()[0,:,0:2]
                gt_np  = dict_list[0]["gt"].cpu().numpy()[0]
                pred_np = dict_list[0][("pred",0)].cpu().numpy()[0]  # (T_pred, N_samples, 2)
                all_obs.append(obs_np)
                all_gt.append(gt_np)
                all_pred.append(pred_np)

                prob_samples = dict_list[0][("prob", 0)]          # tensor, shape: (1, N, T, 3)
                # prob_samples = prob_samples.cpu().numpy() 
                sampled_xy = prob_samples[0, :, :, :2]  # (N_samples, T, 2)

                # -------- 4) 定义区域 & 计算每时刻的区域概率 --------
                # sta
                # CROSSING_POLY_XY = np.array([
                #     [0.3 * W, 0.18 * H],
                #     [0.5 * W, 0.17 * H],
                #     [0.52 * W, 0.34 * H],
                #     [0.35 * W, 0.35 * H],
                # ], dtype=np.float32)

                # small
                # CROSSING_POLY_XY = np.array([
                #     [0.31 * W, 0.19 * H],
                #     [0.49 * W, 0.18 * H],
                #     [0.51 * W, 0.33 * H],
                #     [0.34 * W, 0.34 * H],
                # ], dtype=np.float32)

                # very small
                # CROSSING_POLY_XY = np.array([
                #     [0.32 * W, 0.2 * H],
                #     [0.48 * W, 0.19 * H],
                #     [0.5 * W, 0.32 * H],
                #     [0.37 * W, 0.33 * H],
                # ], dtype=np.float32)

                # only top
                # CROSSING_POLY_XY = np.array([
                #     [0.325 * W, 0.265 * H],
                #     [0.51 * W, 0.255 * H],
                #     [0.52 * W, 0.34 * H],
                #     [0.35 * W, 0.35 * H],
                # ], dtype=np.float32)

                # only bottom
                # CROSSING_POLY_XY = np.array([
                #     [0.3 * W, 0.18 * H],
                #     [0.5 * W, 0.17 * H],
                #     [0.51 * W, 0.255 * H],
                #     [0.325 * W, 0.265 * H],
                # ], dtype=np.float32)

                # only left
                # CROSSING_POLY_XY = np.array([
                #     [0.3 * W, 0.18 * H],
                #     [0.4 * W, 0.175 * H],
                #     [0.435 * W, 0.345 * H],
                #     [0.35 * W, 0.35 * H],
                # ], dtype=np.float32)

                # only right
                CROSSING_POLY_XY = np.array([
                    [0.4 * W, 0.175 * H],
                    [0.5 * W, 0.17 * H],
                    [0.52 * W, 0.34 * H],
                    [0.435 * W, 0.345 * H],
                ], dtype=np.float32)

                # region_prob_seq = compute_region_probability_from_samples_fast(
                #     sampled_xy, CROSSING_POLY_XY
                # )

                # print("区域概率序列 (%):", region_prob_seq * 100)
                ender.record()
                torch.cuda.synchronize()
                curr_run_time = starter.elapsed_time(ender)
                run_times[0].append(curr_run_time)

                # -------- 5) 原来的 density map / metrics / 可视化 --------
                # dict_list = visualizer.prob_to_grid(dict_list)
                result_list.append(metrics(deepcopy(dict_list)))

            #     if visualize:
            #         visualizer(dict_list)
            #     if i == 9:
            #         break

            # visualizer.plot_all_in_one(all_obs, all_gt, all_pred)
            result_info.update(aggregate(result_list))
            print(result_info)

        print(f"execution time: {np.mean(run_times[0]):.2f} " +
              u"\u00B1" + f"{np.std(run_times[0]):.2f} [ms]")
        # print(f"execution time: {np.mean(run_times[1]):.2f} " +
        #       u"\u00B1" + f"{np.std(run_times[1]):.2f} [ms]")
        result_info.update({"execution time": np.mean(
            run_times[0]), "time std": np.std(run_times[0])})


def test(cfg: CfgNode, visualize) -> None:
    data_loader = unified_loader(cfg, rand=False, split="test")
    model = Build_Model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
    evaluate_model(cfg, model, data_loader, visualize)


def aggregate(dict_list: List[Dict]) -> Dict:
    if "nsample" in dict_list[0]:
        ret_dict = {k: np.sum([d[k] for d in dict_list], axis=0) / np.sum(
            [d["nsample"] for d in dict_list]) for k in dict_list[0].keys()}
    else:
        ret_dict = {k: np.mean([d[k] for d in dict_list], axis=0)
                    for k in dict_list[0].keys()}

    return ret_dict


def tune(cfg: CfgNode) -> None:
    import optuna

    def objective_with_arg(cfg):
        _cfg = cfg.clone()
        _cfg.defrost()

        def objective(trial):
            _cfg.MODEL.FLOW.N_BLOCKS = trial.suggest_int(
                "MODEL.FLOW.N_BLOCKS", 1, 3)
            _cfg.MODEL.FLOW.N_HIDDEN = trial.suggest_int(
                "MODEL.FLOW.N_HIDDEN", 1, 3)
            _cfg.MODEL.FLOW.HIDDEN_SIZE = trial.suggest_int(
                "MODEL.FLOW.HIDDEN_SIZE", 32, 128, step=16)
            _cfg.MODEL.FLOW.CONDITIONING_LENGTH = trial.suggest_int(
                "MODEL.FLOW.CONDITIONING_LENGTH", 8, 64, step=8)
            _cfg.SOLVER.LR = trial.suggest_float(
                "SOLVER.LR", 1e-6, 1e-3, log=True)
            _cfg.SOLVER.WEIGHT_DECAY = trial.suggest_float(
                "SOLVER.WEIGHT_DECAY", 1e-12, 1e-5, log=True)

            return train(_cfg, save_model=False)

        return objective

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(sampler=sampler, pruner=pruner,
                                direction='minimize',
                                storage=os.path.join(
                                    "sqlite:///", cfg.OUTPUT_DIR, "optuna.db"),
                                study_name='my_opt',
                                load_if_exists=True)
    study.optimize(objective_with_arg(cfg), n_jobs=4,
                   n_trials=200, gc_after_trial=True)

    trial = study.best_trial

    print(trial.value, trial.params)


def kde(dict_list: List):
    from utils import GaussianKDE
    for data_dict in dict_list:
        for k in list(data_dict.keys()):
            if k[0] == "prob":
                prob = data_dict[k]
                batch_size, _, timesteps, _ = prob.shape
                prob_, gt_traj_log_prob = [], []
                for b in range(batch_size):
                    prob__, gt_traj_prob__ = [], []
                    for i in range(timesteps):
                        kernel = GaussianKDE(prob[b, :, i, :-1])
                        # estimate the prob of predicted future positions for fair comparison of inference time
                        kernel(prob[b, :, i, :-1])
                        prob__.append(deepcopy(kernel))
                        gt_traj_prob__.append(
                            kernel(data_dict["gt"][b, None, i].float()))
                    prob_.append(deepcopy(prob__))
                    gt_traj_log_prob.append(
                        torch.cat(gt_traj_prob__, dim=-1).log())
                gt_traj_log_prob = torch.stack(gt_traj_log_prob, dim=0)
                gt_traj_log_prob = torch.nan_to_num(
                    gt_traj_log_prob, neginf=-10000)
                data_dict[k] = prob_
                data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob

    return dict_list


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg, args.visualize)
    elif args.mode == "tune":
        tune(cfg)


if __name__ == "__main__":
    main()
