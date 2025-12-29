from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
 
# def plot_density(x: np.array, y: np.array, p: np.array, path: Path, traj = None) -> None:
#     plt.pcolormesh(x, y, p.reshape(x.shape),
#                    shading='auto',
#                    cmap=plt.cm.get_cmap("Greens"),
#                    norm=matplotlib.colors.Normalize())
#     if traj is not None:
#         obs, gt = traj
#         sns.lineplot(x=gt[:, 0], y=gt[:, 1],
#                      color='black', marker='o')
#         sns.lineplot(x=obs[:, 0], y=obs[:, 1],
#                      color='green', marker='o')
        
#     plt.axis('off')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()


W, H = 20, 10

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

def plot_density(x, y, p, path: Path, traj=None):
    plt.figure(figsize=(6, 3))

    # ====== 绘制绿色概率密度图 ======
    plt.pcolormesh(
        x, y, p.reshape(x.shape),
        shading='auto',
        cmap=plt.cm.get_cmap("Greens"),
        norm=matplotlib.colors.Normalize(),
        zorder=1
    )

    ax = plt.gca()

    # ====== 绘制 Crossing Polygon 灰色蒙版（不遮挡绿色） ======
    poly = plt.Polygon(
        CROSSING_POLY_XY, 
        color='gray', 
        alpha=0.25,       # 稍微透一点，不遮盖 heatmap
        zorder=2
    )
    ax.add_patch(poly)

    # ====== 绘制轨迹 ======
    # if traj is not None:
    #     obs, gt = traj
    #     sns.lineplot(x=gt[:, 0], y=gt[:, 1], color='green', marker='o', zorder=3)
    #     sns.lineplot(x=obs[:, 0], y=obs[:, 1], color='black', marker='o', zorder=3)

    if traj is not None:
        obs, gt = traj
        sns.lineplot(x=gt[:6, 0], y=gt[:6, 1], color='blue', marker='o', zorder=3)
        sns.lineplot(x=obs[:, 0], y=obs[:, 1], color='black', marker='o', zorder=3)

    # ====== 坐标范围 & 比例 ======
    ax.set_aspect('equal')

    # ====== 坐标刻度 & 网格 ======
    ax.set_xticks(np.linspace(0, W, 11))
    ax.set_yticks(np.linspace(0, H, 11))

    ax.tick_params(axis='both', which='both',
                   color='black',
                   labelsize=8,
                   direction='out',
                   length=4)

    ax.grid(which="major", color="#cccccc", linewidth=0.6)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.3)
    ax.minorticks_on()

    plt.axis('on')

    ax.set_xlim(4, 12)
    ax.set_ylim(0, 6)
    # ax.set_xlim(0, W)
    # ax.set_ylim(0, H)

    plt.savefig(path, dpi=150)
    plt.close()




if __name__ == "__main__":
    from scipy.stats import gaussian_kde

    offsets = [[-2.5, -0.5], [0.0, 1.0]]
    nbins=300
    xi, yi = np.mgrid[-10:10:nbins*1j, -10:10:nbins*1j]
    zi_list = []
    for offset in offsets:
        # create data
        x = np.random.normal(size=500, scale=0.5) + offset[0]
        y = np.random.normal(size=500, scale=0.5) + offset[1]
        
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = gaussian_kde([x,y])
        zi_list.append(k(np.vstack([xi.flatten(), yi.flatten()])))

    z = np.zeros_like(zi_list[0])
    for i, zi in enumerate(zi_list):
        z += zi
        plot_density(xi, yi, z/(i+1), path=f"{i}.png")
    

