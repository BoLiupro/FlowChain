from yacs.config import CfgNode
from typing import Dict, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from joblib import delayed, Parallel
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

from visualization.density_plot import plot_density
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import griddata, RBFInterpolator, interp2d
from scipy.stats import multivariate_normal

W, H = 20, 10

class Visualizer(ABC):
    def __init__(self, cfg: CfgNode):
        pass

    @abstractmethod
    def __call__(self, dict_list: List[Dict]) -> None:
        pass

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()


class TP_Visualizer(Visualizer):
    def __init__(self, cfg: CfgNode):
        self.model_name = cfg.MODEL.TYPE

        self.output_dir = Path(cfg.OUTPUT_DIR) / "visualize"
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = cfg.DATA.DATASET_NAME

        import dill
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / \
            "processed_data" / f"{cfg.DATA.DATASET_NAME}_test.pkl"
        with open(env_path, 'rb') as f:
            self.env = dill.load(f, encoding='latin1')

        # TODO: params for other nodes
        from data.TP.trajectron_dataset import hypers
        node = 'PEDESTRIAN'
        self.state = hypers[cfg.DATA.TP.PRED_STATE][node]
        mean, std = self.env.get_standardize_params(self.state, node)
        self.mean = mean
        self.std = std
        self.std = self.env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]

        self.num_grid = 100
        if hasattr(self.env, "gt_dist"):
            self.gt_dist = self.env.gt_dist
        else:
            self.gt_dist = None

        self.observe_length = cfg.DATA.OBSERVE_LENGTH

    def __call__(self, dict_list: List[Dict]) -> None:
        index = dict_list[0]['index']
        min_pos, max_pos = self.get_minmax(index)

        # (batch, timesteps, [x,y])
        obs = self.to_numpy(dict_list[0]['obs'][:, :, 0:2])
        gt = self.to_numpy(dict_list[0]['gt'])

        pred = []
        for d in dict_list:
            pred.append(self.to_numpy(d[("pred", 0)][:, :, None]))
            assert np.all(obs == self.to_numpy(d["obs"][:, :, 0:2]))
            assert np.all(gt == self.to_numpy(d["gt"]))

        # (batch, timesteps, num_trials, [x,y])
        pred = np.concatenate(pred, axis=2)
        for i in range(len(obs)):
            self.plot2d_trajectories(obs[i:i+1],
                                     gt[i:i+1],
                                     pred[i:i+1],
                                     index[i],
                                     max_pos,
                                     min_pos)

        if ("prob", 0) in dict_list[0]:
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)

            xx, yy = self.get_grid(index)

            for k in dict_list[0].keys():
                if k[0] == "prob":
                    update_step = k[1]
                    prob = dict_list[0][k]

                    bs, _, timesteps = prob.shape

                    obs = self.to_numpy(dict_list[0]['obs'])[..., :2]
                    gt = self.to_numpy(dict_list[0]['gt'])
                    traj = np.concatenate([obs, gt], axis=1)
                    obs = traj[:, :self.observe_length + update_step]
                    gt = traj[:, self.observe_length + update_step-1:]

                    zz_list = []
                    # for j in range(timesteps):
                    draw_steps = min(timesteps, 5)   # 最多画 8 步
                    for j in range(draw_steps):
                        zz = prob[0, :, j].reshape(xx.shape)
                        zz /= np.max(zz)
                        # plot_density(xx, yy, zz, path=path_density_map / f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_{j}.png",
                        #              traj=[obs[i], gt[i]])
                        zz_list.append(zz)

                    zz_sum = sum(zz_list)
                    plot_density(xx, yy, zz_sum,
                                 path=path_density_map /
                                 f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_sum.png",
                                 traj=[obs[i], gt[i]])

    def prob_to_grid(self, dict_list: List[Dict]) -> List:
        if ("prob", 0) in dict_list[0]:
            index = dict_list[0]['index']
            # min_pos, max_pos = self.get_minmax(index)
            # 改为固定尺度映射

            min_pos = np.array([0,0])
            max_pos = np.array([W,H])
            xx, yy = self.get_grid(index)

            for data_dict in dict_list:
                data_dict["grid"] = [xx, yy]
                data_dict["minmax"] = [min_pos, max_pos]
                for k in list(data_dict.keys()):
                    if k[0] == "prob":
                        prob = data_dict[k]
                        if type(prob) == torch.Tensor:
                            prob = self.to_numpy(prob)
                            batch, _, timesteps, _ = prob.shape

                            zz_batch = []
                            for i in range(batch):
                                zz_timesteps = Parallel(n_jobs=timesteps)(delayed(self.griddata_on_cluster)(i, prob, xx, yy, max_pos, min_pos, j)
                                                                          for j in range(timesteps))
                                #zz_timesteps = [self.griddata_on_cluster(i, prob, xx, yy, max_pos, min_pos, j) for j in range(timesteps)]
                                zz_timesteps = np.stack(
                                    zz_timesteps, axis=-1).reshape(-1, timesteps)
                                zz_batch.append(zz_timesteps)

                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch

                        elif self.model_name == "Trajectron" or self.model_name == "GT_Dist":
                            value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[
                                None, :, None].tile(1, 1, prob.mus.shape[2], 1).cuda()
                            zz_batch = torch.exp(prob.log_prob(value))
                            data_dict[k] = self.to_numpy(zz_batch)

                        else:
                            zz_batch = []
                            for i in range(len(prob)):
                                zz_timesteps = []
                                for kernel in prob[i]:
                                    zz = kernel(torch.Tensor(
                                        np.array([xx.flatten(), yy.flatten()]).T).cuda())
                                    zz_timesteps.append(zz.cpu().numpy())
                                zz_timesteps = np.stack(zz_timesteps, axis=1)
                                zz_batch.append(zz_timesteps)
                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch

                        if ("gt_traj_log_prob", k[1]) not in data_dict:
                            gt = data_dict["gt"][:, k[1]:].cpu()
                            timesteps = gt.shape[1]
                            gt_traj_prob = np.array([interp2d(xx, yy, zz_batch[:, :, t])(
                                gt[:, t, 0], gt[:, t, 1]) for t in range(timesteps)])
                            gt_traj_log_prob = torch.log(
                                torch.Tensor(gt_traj_prob)).squeeze()
                            if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                                mask = torch.isnan(
                                    gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
                                value = torch.min(gt_traj_log_prob[~mask])
                                gt_traj_log_prob = torch.nan_to_num(
                                    gt_traj_log_prob, nan=value, neginf=value)
                            data_dict[("gt_traj_log_prob", k[1])
                                      ] = gt_traj_log_prob[None]

                if self.gt_dist is not None:  # assume simfork
                    bs, timesteps, d = data_dict["gt"].shape
                    split = True
                    value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[
                        None, :, None].tile(1, 1, timesteps, 1).cuda()
                    if split:
                        gaussians = []
                        for i in range(len(self.gt_dist)):
                            gt_base_traj = torch.Tensor(
                                self.gt_dist[[i] * bs, -timesteps:]).cuda()
                            gaussians.append(
                                Normal(gt_base_traj[..., :2], gt_base_traj[..., 2:]))
                        data_dict["gt_prob"] = sum(
                            [torch.exp(g.log_prob(value).sum(dim=-1)) for g in gaussians])
                    else:
                        gt_base_traj = torch.Tensor(
                            self.gt_dist[data_dict["gt"][:, -1, 1] < 0, -timesteps:]).cuda()
                        gaussian = Normal(
                            gt_base_traj[:, :2], gt_base_traj[:, 2:])
                        data_dict["gt_prob"] = torch.exp(
                            gaussian.log_prob(value).sum(dim=-1))

        # # 在 prob_to_grid(dict_list) 最后 return 前加入：

        # for data_dict in dict_list:
        #     data_dict["obs_array"] = data_dict["obs"][..., :2].cpu().numpy()
        #     data_dict["gt_array"] = data_dict["gt"][..., :2].cpu().numpy()

        #     if ('pred', 0) in data_dict:
        #         data_dict["pred_array"] = data_dict[('pred', 0)][..., :2].cpu().numpy()
        #     else:
        #         data_dict["pred_array"] = None

        return dict_list

    # def get_grid(self, index):
    #     min_pos, max_pos = self.get_minmax(index)
    #     xs = np.linspace(min_pos[0], max_pos[0], num=self.num_grid)
    #     ys = np.linspace(min_pos[1], max_pos[1], num=self.num_grid)
    #     xx, yy = np.meshgrid(xs, ys)

    #     return xx, yy

    def get_grid(self, index):
        # 不再用 min_pos / max_pos
        W, H = 20, 10   # 你原来用的缩放
        xs = np.linspace(0, W, num=self.num_grid)
        ys = np.linspace(0, H, num=self.num_grid)
        xx, yy = np.meshgrid(xs, ys)
        return xx, yy


    def get_minmax(self, index):
        idx = [s.name for s in self.env.scenes].index(index[0][0])
        max_pos, min_pos = self.env.scenes[idx].calculate_pos_min_max()
        max_pos += 0.05 * (max_pos - min_pos)
        min_pos -= 0.05 * (max_pos - min_pos)
        return min_pos, max_pos

    def griddata_on_cluster(self, i, prob, xx, yy, max_pos, min_pos, j):
        prob_ = prob[i, :, j]
        prob_ = prob_[np.where(np.isinf(prob_).sum(axis=1) == 0)]
        prob_ = prob_[np.where(np.isnan(prob_).sum(axis=1) == 0)]
        lnk = linkage(prob_[:, :-1],
                      method='single',
                      metric='euclidean')
        idx_cls = fcluster(lnk, t=np.linalg.norm(max_pos-min_pos)*0.003,
                           criterion='distance')
        idx_cls -= 1

        zz_ = []
        for c in range(np.max(idx_cls)+1):
            try:
                zz_.append(griddata(prob_[idx_cls == c, :-1],
                                    prob_[idx_cls == c, -1],
                                    (xx, yy), method='linear',
                                    fill_value=0.0)
                           )
            except:
                pass

        zz = sum(zz_)

        intp = RBFInterpolator(np.array([xx, yy]).reshape(
            2, -1).T, zz.flatten(), smoothing=10, kernel='linear', neighbors=8)
        zz = intp(np.array([xx, yy]).reshape(2, -1).T).reshape(xx.shape)

        prob_ = prob[i, :, j, -1]
        prob_ = prob_[~np.isnan(prob_)]
        zz = np.clip(zz, a_min=0.0, a_max=np.percentile(prob_, 95))
        return zz

    def plot2d_trajectories(self,
                            obs:  np.ndarray,
                            gt:   np.ndarray,
                            pred: np.ndarray,
                            index: Tuple,
                            max_pos: np.ndarray,
                            min_pos: np.ndarray) -> None:
        """plot 2d trajectories

        Args:
            obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
            gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
            pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
            img_path (Path): Path
        """

        # N_seqs, N_timesteps, N_trials, N_dim = pred.shape
        # gt_vis = np.zeros([N_seqs, N_timesteps+1, N_dim])
        # gt_vis[:, 0] = obs[:, -1]
        # gt_vis[:, 1:] = gt

        # pred_vis = np.zeros([N_seqs, N_timesteps+1, N_trials, N_dim])
        # # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
        # pred_vis[:, 0] = obs[:, -1][:, None]
        # pred_vis[:, 1:] = pred



        # f, ax = plt.subplots(1, 1)
        # ax.set_aspect('equal', adjustable='box')
        # # ax.set_xlim(min_pos[0], max_pos[0])
        # # ax.set_ylim(min_pos[1], max_pos[1])
        # ax.set_xlim(0, W)
        # ax.set_ylim(0, H)

        # for j in range(N_seqs):
        #     sns.lineplot(x=obs[j, :, 0], y=obs[j, :, 1], color='black',
        #                  legend='brief', label="obs", marker='o')
        #     sns.lineplot(x=gt_vis[j, :, 0], y=gt_vis[j, :, 1],
        #                  color='blue', legend='brief', label="GT", marker='o')
        #     for i in range(pred.shape[2]):
        #         if i == 0:
        #             sns.lineplot(x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1],
        #                          color='green', legend='brief', label="pred", marker='o')
        #         else:
        #             sns.lineplot(x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1], color='green', marker='o')


        # 原 shape：pred: (N_seqs, T_pred, N_trials, 2)
        N_seqs, N_timesteps, N_trials, N_dim = pred.shape

        # ============= 重新构造 gt_vis/pred_vis（只保留前5步） =============
        draw_T = min(12, N_timesteps)   # 最多画5步

        # obs 全部步骤正常画
        # 改 gt：只画前5步
        gt_vis = np.zeros([N_seqs, draw_T + 1, N_dim])
        gt_vis[:, 0] = obs[:, -1]
        gt_vis[:, 1:] = gt[:, :draw_T]

        # 改 pred：只画前5步
        pred_vis = np.zeros([N_seqs, draw_T + 1, N_trials, N_dim])
        pred_vis[:, 0] = obs[:, -1][:, None]
        pred_vis[:, 1:] = pred[:, :draw_T]

        # ====== 计算平均轨迹（Expectation / Mean trajectory） ======
        # pred shape: (N_seqs, T_pred, N_trials, 2)

        mean_pred = pred.mean(axis=2)   # (N_seqs, T_pred, 2)

        mean_pred_vis = np.zeros([N_seqs, draw_T + 1, N_dim])
        mean_pred_vis[:, 0] = obs[:, -1]
        mean_pred_vis[:, 1:] = mean_pred[:, :draw_T]


        # ============= 绘图 =============
        f, ax = plt.subplots(1, 1)

        # ============= 区域定义 =============
        CROSSING_POLY = np.array([
            [0.3 * W, 0.18 * H],
            [0.5 * W, 0.17 * H],
            [0.52 * W, 0.34 * H],
            [0.35 * W, 0.35 * H],
        ], dtype=np.float32)

        # ============= 区域蒙版 =============
        poly = plt.Polygon(
            CROSSING_POLY,
            color='gray',
            alpha=0.25,
            zorder=1   # 一定要比轨迹低
        )
        ax.add_patch(poly)

        ax.set_aspect('equal')
        # ax.set_xlim(0, W)
        # ax.set_ylim(0, H)
        ax.set_xlim(4, 12)
        ax.set_ylim(0, 6)

        for j in range(N_seqs):

            # ------ gt：只画前5步 ------
            sns.lineplot(
                x=gt_vis[j, :, 0],
                y=gt_vis[j, :, 1],
                color='blue', label="GT", marker='o'
            )

            # ------ pred：只画前5步 ------
            # for i in range(N_trials):
            #     if i == 0:
            #         sns.lineplot(
            #             x=pred_vis[j, :, i, 0],
            #             y=pred_vis[j, :, i, 1],
            #             color='green', marker='o', label='pred'
            #         )
            #         # continue
            #     else:
            #         sns.lineplot(
            #             x=pred_vis[j, :, i, 0],
            #             y=pred_vis[j, :, i, 1],
            #             color='green', marker='o'
            #         )
            for i in range(N_trials):
                if i == 0:
                    sns.lineplot(
                        x=mean_pred_vis[j, :, 0],
                        y=mean_pred_vis[j, :, 1],
                        color='green', marker='o', label='pred'
                    )
                    # continue
                else:
                    sns.lineplot(
                        x=mean_pred_vis[j, :, 0],
                        y=mean_pred_vis[j, :, 1],
                        color='green', marker='o'
                    )

                        # ------ obs：全步骤 ------
            sns.lineplot(
                x=obs[j, :, 0], 
                y=obs[j, :, 1],
                color='black', label="obs", marker='o'
            )
        img_path = self.output_dir / \
            f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}.png"
        plt.savefig(img_path)
        plt.close()

        # 保存obs,gt_vis,pred_vis为csv文件
        # np.savetxt(self.output_dir / f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}_obs.csv",
        #            obs[0], delimiter=',')
        # np.savetxt(self.output_dir / f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}_gt.csv",
        #            gt_vis[0], delimiter=',')
        # np.savetxt(self.output_dir / f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}_pred.csv",
        #            pred_vis[0].reshape(N_timesteps+1, N_trials*N_dim), delimiter=',')

    def plot_all_in_one(self, obs_all, gt_all, pred_all):
        """
        obs_all: list of obs arrays, each (T_obs, 2)
        gt_all:  list of gt arrays, each (T_pred, 2)
        pred_all: list of pred arrays, each (T_pred, N_samples, 2) or (T_pred, 2)
        """

        # ====================================================
        # 统一 pred_all 维度：确保 (T_pred, N_samples, 2)
        # ====================================================
        for i in range(len(pred_all)):
            pred = pred_all[i]

            # (T_pred,2) → (T_pred,1,2)
            if pred.ndim == 2:
                pred = pred[:, None, :]    # N_samples = 1

            pred_all[i] = pred

        # ====================================================
        # 正式绘图部分
        # ====================================================
        f, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.set_aspect('equal')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)

        CROSSING_POLY = np.array([
            [0.3 * W, 0.18 * H],
            [0.5 * W, 0.17 * H],
            [0.52 * W, 0.34 * H],
            [0.35 * W, 0.35 * H],
        ], dtype=np.float32)

        # 区域蒙版
        poly = plt.Polygon(
            CROSSING_POLY,
            color='gray',
            alpha=0.25,
            zorder=2
        )
        ax.add_patch(poly)

        # ====================================================
        # 逐个轨迹叠加
        # ====================================================
        is_first = True

        for obs, gt, pred in zip(obs_all, gt_all, pred_all):

            # obs
            sns.lineplot(
                x=obs[:, 0], y=obs[:, 1],
                color='black', marker='o',
                label="obs" if is_first else None
            )

            # gt 只画前5步
            draw_T = min(5, len(gt))
            gt_draw = np.zeros((draw_T + 1, 2))
            gt_draw[0] = obs[-1]
            gt_draw[1:] = gt[:draw_T]

            sns.lineplot(
                x=gt_draw[:, 0], y=gt_draw[:, 1],
                color='blue', marker='o',
                label="gt" if is_first else None
            )

            # pred：统一后是 (T_pred, N_samples, 2)
            T_pred, N_samples, _ = pred.shape
            draw_T = min(5, T_pred)

            pred_draw = np.zeros((draw_T + 1, N_samples, 2))
            pred_draw[0] = obs[-1]
            pred_draw[1:] = pred[:draw_T]

            for s in range(N_samples):
                sns.lineplot(
                    x=pred_draw[:, s, 0],
                    y=pred_draw[:, s, 1],
                    color='green',
                    marker='o',
                    alpha=0.6,
                    label='pred' if (is_first and s == 0) else None
                )

            is_first = False

        ax.set_title("All Trajectories")
        out_path = self.output_dir / "all_in_one.png"
        plt.savefig(out_path, dpi=150)
        plt.close()



def plot2d_trajectories_samples(
        obs:  np.ndarray,
        gt:   np.ndarray,
        max_pos: np.ndarray,
        min_pos: np.ndarray) -> None:
    """plot 2d trajectories

    Args:
        obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
        gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
        pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
        img_path (Path): Path
    """

    N_seqs = len(gt)
    _, N_timesteps, N_dim = gt[0].shape

    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xlim(min_pos[0], max_pos[0])
    # ax.set_ylim(min_pos[1], max_pos[1])
    plt.xlim(5,12)
    plt.ylim(0,5)

    for j in range(N_seqs):
        gt_vis = np.zeros([N_timesteps+1, N_dim])
        gt_vis[0] = obs[j][0, -1]
        gt_vis[1:] = gt[j][0]
        if j == 0:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                         legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                         color='black', legend='brief', label="GT", marker='o')
        else:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                         marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                         color='black', marker='o')

    img_path = "gts.png"
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
