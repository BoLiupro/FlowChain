import os
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

W, H = 20, 10  # 缩放比例

# 定义路口 polygon

CROSSING_POLY = np.array([
    [0.3 * W, 0.18 * H],
    [0.5 * W, 0.17 * H],
    [0.52 * W, 0.34 * H],
    [0.35 * W, 0.35 * H]
])

CAR_DETECTION_POLY = Polygon([
    [0.2 * W, 0.18 * H],
    [0.9 * W, 0.17 * H],
    [0.9 * W, 0.34 * H],
    [0.25 * W, 0.35 * H]
])


# ============================================================
# 1. 读取 ultralytics 输出
# ============================================================
def load_ultralytics(folder):
    """
    返回 df: frame_id, track_id, x, y, cls, xc, yc, w, h
    行人：直接记录点
    车辆：记录 bbox 与中心点，车头之后另算
    """
    records = []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue

        frame_id = int("".join(filter(str.isdigit, fname)))

        with open(os.path.join(folder, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                cls, xc, yc, w, h, tid = parts
                tid = int(float(tid))
                xc, yc, w, h = map(float, (xc, yc, w, h))

                if cls == "0":  # 行人
                    x = xc * W
                    y = H - (yc + h / 2) * H
                    records.append([frame_id, tid, x, y, 0, xc, yc, w, h])

                elif cls == "2":  # 车辆
                    x_center = xc * W
                    y_center = H - (yc + h / 2) * H
                    records.append([frame_id, tid, x_center, y_center, 2, xc, yc, w, h])

    df = pd.DataFrame(records,
                      columns=["frame_id", "track_id", "x", "y", "cls", "xc", "yc", "w", "h"])
    df.sort_values(["track_id", "frame_id"], inplace=True)
    return df


# ============================================================
# 2. 重采样到 2.5Hz
# ============================================================
def resample_to_2_5hz(df):
    df_list = []
    for tid, g in df.groupby("track_id"):
        sampled = g[g.frame_id % 10 == 1].copy()
        if len(sampled) > 0:
            df_list.append(sampled)
    return pd.concat(df_list).reset_index(drop=True)


# ============================================================
# 3. 智能插值（带外推）
# ============================================================
def smart_extrapolate(series):
    series_interp = series.interpolate(method="linear")

    first_valid = series_interp.first_valid_index()
    if first_valid is not None and first_valid > series_interp.index[0]:
        slope = 0
        if first_valid + 1 in series_interp.index:
            slope = series_interp[first_valid + 1] - series_interp[first_valid]
        for i in range(first_valid - 1, series_interp.index[0] - 1, -1):
            series_interp[i] = series_interp[i + 1] - slope

    last_valid = series_interp.last_valid_index()
    if last_valid is not None and last_valid < series_interp.index[-1]:
        slope = 0
        if last_valid - 1 in series_interp.index:
            slope = series_interp[last_valid] - series_interp[last_valid - 1]
        for i in range(last_valid + 1, series_interp.index[-1] + 1):
            series_interp[i] = series_interp[i - 1] + slope

    return series_interp


def interpolate_track(df_track, target_frames):
    df = df_track.set_index("frame_id")
    df = df.reindex(target_frames)

    if df[["x", "y"]].isna().all().all():
        return None

    df["x"] = smart_extrapolate(df["x"])
    df["y"] = smart_extrapolate(df["y"])

    df["track_id"] = df_track["track_id"].iloc[0]
    # df["cls"] = df_track["cls"].iloc[0]
    # df["xc"] = df_track["xc"].iloc[0]
    # df["yc"] = df_track["yc"].iloc[0]
    # df["w"] = df_track["w"].iloc[0]
    # df["h"] = df_track["h"].iloc[0]

    return df.reset_index()

from scipy.signal import savgol_filter

def smooth_tracks(df, window=5, poly=2):
    """
    对轨迹使用 Savitzky–Golay 滤波器做平滑处理
    window: 必须为奇数
    """
    if df.empty:
        return df
    df_list = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_id")
        if len(g) >= window:
            g["x"] = savgol_filter(g["x"], window, poly)
            g["y"] = savgol_filter(g["y"], window, poly)
        df_list.append(g)
    return pd.concat(df_list).reset_index(drop=True)


# ============================================================
# 车辆运动方向推断：判断车头
# ============================================================
def determine_vehicle_head(df_tid):
    vx = df_tid["x"].diff().mean()
    vy = df_tid["y"].diff().mean()

    row = df_tid.iloc[-1]
    xc, yc, w, h = row["xc"], row["yc"], row["w"], row["h"]

    x_left  = xc - w/2
    x_right = xc + w/2
    y_car   = 1 - (yc + h/2)

    if abs(vx) > abs(vy): 
        if vx > 0:
            return x_right, y_car
        elif vx < 0:
            return x_left, y_car

    return (x_left + x_right) / 2, y_car


# ============================================================
# 判断是否静止
# ============================================================
def is_stationary(df_tid, thr=0.1):
    if len(df_tid) < 2:
        return True
    vx = np.diff(df_tid.x)
    vy = np.diff(df_tid.y)
    speed = np.sqrt(vx**2 + vy**2)
    return np.max(speed) < thr

def vehicle_enters_crossing(df_tid, obs_frames, center_frame):
    """
    只检查 obs_frames + center_frame 是否进入 CAR_DETECTION_POLY
    """
    target_frames = set(obs_frames + [center_frame])

    for _, row in df_tid.iterrows():
        if row.frame_id in target_frames:
            if Point(row.x, row.y).within(CAR_DETECTION_POLY):
                return True

    return False



# ============================================================
# 4. 构造 obs+pred
# ============================================================
def build_observe_predict(df, center_frame, target_tid=None):

    STEP = 10
    OBS_FRAMES = 8
    PRED_FRAMES = 12

    obs_frames = [center_frame - STEP*i for i in range(OBS_FRAMES, 0, -1)]
    pred_frames = [center_frame + STEP*i for i in range(PRED_FRAMES)]
    target_frames = obs_frames + pred_frames

    active_tids = df[df.frame_id == center_frame]["track_id"].unique()
    
    if target_tid is not None:
        if target_tid in active_tids:
            active_tids = [target_tid]
        else:
            return pd.DataFrame()

    all_rows = []

    for tid in active_tids:
        df_tid = df[df.track_id == tid].copy()

        # ---------- 车辆检查逻辑 ----------
        if df_tid.cls.iloc[0] == 2:

            # head = determine_vehicle_head(df_tid)
            # if head is None:
            #     continue

            # df_tid["x"], df_tid["y"] = head
            df_tid["x"], df_tid["y"] = df_tid["x"], df_tid["y"]


            if vehicle_enters_crossing(df_tid, obs_frames, center_frame) and not is_stationary(df_tid):
                print(f"[警告] 车辆 {tid} 在 crossing 区域内运动，程序终止")
                sys.exit(0)
                # return pd.DataFrame() # Return empty DF to signal skip


            # 车辆不进入输出数据
            continue

        # ---------- 行人正常插值 ----------
        df_full = interpolate_track(df_tid, target_frames)
        if df_full is not None:
            all_rows.append(df_full[["frame_id", "track_id", "x", "y"]])

    if len(all_rows) == 0:
        return pd.DataFrame()

    return pd.concat(all_rows)


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":

    folder = "/root/workspace/FlowChain-ICCV2023/red_data/red_full"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--center_frame', type=int, default=81, help='Center frame for processing')
    parser.add_argument('--output_dir', type=str, default="/root/workspace/FlowChain-ICCV2023/src/data/TP/raw_data/zara2", help='Output directory')
    parser.add_argument('--track_id', type=int, default=None, help='Specific track ID to process')
    args = parser.parse_args()
    center_frame = args.center_frame
    out_dir = args.output_dir
    track_id = args.track_id

    df_raw = load_ultralytics(folder)
    df_2_5hz = resample_to_2_5hz(df_raw)
    df_final = build_observe_predict(df_2_5hz, center_frame, target_tid=track_id)
    df_final = smooth_tracks(df_final)
    
    if df_final.empty:
        print("No data generated.")
        # Create empty file to overwrite previous data
        os.makedirs(f"{out_dir}/train", exist_ok=True)
        with open(f"{out_dir}/train/converted_tracks.txt", 'w') as f:
            pass
        sys.exit(0)

    df_final = df_final.sort_values(["frame_id", "track_id"])

    df_final.to_csv(f"{out_dir}/train/converted_tracks.txt", sep="\t", index=False, header=False)
    df_final.to_csv(f"{out_dir}/test/converted_tracks.txt", sep="\t", index=False, header=False)
    df_final.to_csv(f"{out_dir}/val/converted_tracks.txt", sep="\t", index=False, header=False)

    print("已保存：converted_tracks.txt")
    print(df_final)
