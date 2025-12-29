import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

ROOT = "plot"     # 你的输出目录
OBS_DIR  = os.path.join(ROOT, "obs")
GT_DIR   = os.path.join(ROOT, "gt")
PRED_DIR = os.path.join(ROOT, "pred")

W, H = 20, 10  # 你的坐标转换比例

CROSSING_POLY = np.array([
    [0.3 * W, 0.18 * H],
    [0.5 * W, 0.17 * H],
    [0.52 * W, 0.34 * H],
    [0.35 * W, 0.35 * H]
], dtype=np.float32)
CROSSING_POLYGON = Polygon(CROSSING_POLY)

OBS_LEN = 8
PRED_LEN = 12
K_EVALUATE = 5


def load_traj(path):
    return np.loadtxt(path)


def load_multi_agent_traj(arr, agent_len):
    """
    将堆叠的多个目标轨迹按长度切分
    arr.shape = [N * T, 2]
    返回 list，每个元素 shape = [T, 2]
    """
    num_agents = arr.shape[0] // agent_len
    agents = []
    for i in range(num_agents):
        seg = arr[i * agent_len:(i + 1) * agent_len]
        agents.append(seg)
    return agents


def get_idx_list(folder):
    files = sorted(os.listdir(folder))
    idx = []
    for f in files:
        if "_" not in f:
            continue
        i = int(f.split("_")[1].split(".")[0])
        idx.append(i)
    return sorted(idx)


def in_crossing(pt):
    return CROSSING_POLYGON.contains(Point(pt[0], pt[1]))


def plot_crossing_region():
    poly = np.vstack([CROSSING_POLY, CROSSING_POLY[0]])
    plt.plot(poly[:, 0], poly[:, 1], "r--", lw=2, label="Crossing Region")

def plot_single_agent(agent_id, obs, gt, pred_mean, K_EVALUATE=5, save_dir="/root/workspace/FlowChain/plot/vis"):
    """
    单独为每个 agent 生成独立 PNG 可视化（OBS / GT / PRED）
    坐标范围根据该 agent 的轨迹自动缩放
    """

    # 拼接 obs → gt[0]，保持你的风格
    obs_full = np.vstack([obs[:, :2], gt[0, :]])

    # 轨迹点集合，用于自动调整坐标轴
    all_pts = np.vstack([obs_full, gt[:K_EVALUATE], pred_mean[:K_EVALUATE]])

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    # 给画面留一点边缘（更舒服）
    margin_x = (x_max - x_min) * 0.2 + 1e-3
    margin_y = (y_max - y_min) * 0.2 + 1e-3

    plt.figure(figsize=(6, 5))

    # 曲线
    plt.plot(obs_full[:, 0], obs_full[:, 1], "bo-", markersize=3, alpha=0.8, label="OBS")
    plt.plot(gt[:K_EVALUATE, 0], gt[:K_EVALUATE, 1], "go-", markersize=3, alpha=0.8, label="GT")
    plt.plot(pred_mean[:K_EVALUATE, 0], pred_mean[:K_EVALUATE, 1], "ro-", markersize=3, alpha=0.8, label="PRED MEAN")

    # crossing 区域（全局一样）
    plot_crossing_region()

    plt.title(f"Agent {agent_id} Trajectory (OBS / GT / PRED)")
    plt.legend()

    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)
    plt.gca().set_aspect("equal", "box")

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"agent_{agent_id}.png")
    plt.savefig(fname)
    plt.close()

    print(f"单独图已保存: {fname}")


# ============================================================
# 修改 analyze()，在你的循环中调用 plot_single_agent()
# ============================================================

def analyze():

    idx_list = get_idx_list(PRED_DIR)
    N_SAMPLE = len(idx_list)
    print(f"检测到 {N_SAMPLE} 份 pred 采样数据")

    # ===============================
    # 加载 obs / gt（第一份即可）
    # ===============================
    obs_raw = load_traj(os.path.join(OBS_DIR, f"obs_{idx_list[0]}.txt"))
    gt_raw  = load_traj(os.path.join(GT_DIR,  f"gt_{idx_list[0]}.txt"))

    obs_agents = load_multi_agent_traj(obs_raw, OBS_LEN)
    gt_agents  = load_multi_agent_traj(gt_raw, PRED_LEN)

    num_agents = len(obs_agents)
    print(f"检测到 {num_agents} 个目标")

    # ===============================
    # 加载所有 pred
    # ===============================
    pred_all = []
    for idx in idx_list:
        p_raw = load_traj(os.path.join(PRED_DIR, f"pred_{idx}.txt"))
        p_agents = load_multi_agent_traj(p_raw, PRED_LEN)
        pred_all.append(p_agents)

    pred_all = np.array(pred_all)   # [K, num_agents, 12, 2]
    print("pred_all shape =", pred_all.shape)

    # ======================================================
    # 多目标总体可视化（保持你原来的）
    # ======================================================
    plt.figure(figsize=(8, 6))

    for a in range(num_agents):
    # for a in range(8):
    
        obs = obs_agents[a]
        gt  = gt_agents[a]
        pred_mean = pred_all[:, a].mean(axis=0)

        obs_full = np.vstack([obs[:, :2], gt[0, :]])
        plt.plot(obs_full[:, 0], obs_full[:, 1], "bo-", markersize=3, alpha=0.8)
        plt.plot(gt[:K_EVALUATE, 0], gt[:K_EVALUATE, 1], "go-", markersize=3, alpha=0.8)
        plt.plot(pred_mean[:K_EVALUATE, 0], pred_mean[:K_EVALUATE, 1], "ro-", markersize=3, alpha=0.8)

        final_x, final_y = gt[min(K_EVALUATE - 3, pred_mean.shape[0] - 1)]
        plt.text(final_x, final_y, f"Agent {a}", fontsize=10)

        # ============ 新增：单独绘制每个 agent ============
        # plot_single_agent(a, obs, gt, pred_mean, K_EVALUATE)

    plot_crossing_region()
    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.xlim(5,12)
    plt.ylim(0,5)
    plt.gca().set_aspect("equal", "box")

    plt.title("All Agents Trajectories (OBS / GT / PRED_MEAN)")
    plt.legend(["OBS", "GT", "PRED MEAN"])
    plt.savefig("/root/workspace/FlowChain-ICCV2023/plot/vis/all_agents.png")
    plt.close()

    # ======================================================
    # crossing 概率
    # ======================================================
    crossing_prob = np.zeros((num_agents, K_EVALUATE))

    print("\n=== 每个目标前5步 crossing 概率 ===\n")

    for a in range(num_agents):
        for t in range(K_EVALUATE):
            hits = 0
            for k in range(N_SAMPLE):
                if in_crossing(pred_all[k, a, t]):
                    hits += 1
            crossing_prob[a, t] = hits / N_SAMPLE

        print(f"Agent {a}: {crossing_prob[a]}")

    np.savetxt("/root/workspace/FlowChain-ICCV2023/plot/crossing_prob_per_agent.txt",
               crossing_prob, fmt="%.6f")
    print("\nCrossing 概率已保存到 crossing_prob_per_agent.txt")


if __name__ == "__main__":
    analyze()
