"""
基于 MediaPipe PoseLandmarker 骨架检测的智能视频剪辑模块

功能：分析视频中人物的骨架，根据6条规则标记"坏帧"，
     找出最连贯的骨架片段，输出裁剪后的视频。
     v2 新增：场景切换检测 + 手部动作增强

安装依赖: pip install mediapipe opencv-python numpy moviepy
"""

import os
import sys
import math
import shutil
import argparse
import tempfile
import numpy as np
import cv2

# ── MediaPipe 懒加载 ──────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None
    mp_tasks = None
    mp_vision = None
    BaseOptions = None

# ── 骨架关键点索引 ────────────────────────────────────────────────
# MediaPipe Pose 33 landmarks
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28

# ── 保留规则（姿态1评分）参数 ──────────────────────────────────
# 归一化坐标范围（以双髋中心为原点，肩宽为缩放单位）
# 格式: {关键点索引: ((x_min, x_max), (y_min, y_max)), ...}
POSE_RANGES = {
    0:  ((-1.927, 1.731), (-4.168, -0.768)),  # 鼻
    11: ((-1.304, 1.936), (-3.072, -0.561)),  # 左肩
    12: ((-2.167, 0.978), (-3.096, -0.526)),  # 右肩
    13: ((-0.960, 2.037), (-2.066, 0.096)),  # 左肘
    14: ((-2.170, 0.657), (-2.048, 0.123)),  # 右肘
    15: ((-0.942, 2.119), (-2.080, 0.896)),  # 左腕
    16: ((-2.203, 0.829), (-2.110, 0.752)),  # 右腕
    23: ((0.157, 0.385), (-0.073, 0.060)),   # 左髋
    24: ((-0.385, -0.157), (-0.060, 0.073)), # 右髋
    25: ((-0.314, 1.065), (0.236, 2.353)),   # 左膝
    26: ((-0.869, 0.462), (0.277, 2.314)),   # 右膝
}

# 关节角度参考值（来自好素材骨架特征）
POSE_ELBOW_ANGLE = 124.5   # 肩-肘-腕夹角（度）
POSE_KNEE_ANGLE = 164.9    # 髋-膝-踝夹角（度）
POSE_ANGLE_TOLERANCE = 20.0  # 角度容差（度）
# 距离阈值（放宽到2.5以匹配更宽的数据范围）
POSE_DISTANCE_THRESHOLD = 2.5

# ── 默认参数 ──────────────────────────────────────────────────────
_DEFAULT_CONFIG = {
    "fold_angle_threshold": 30,        # 关节角度小于此值判定为折叠（度）
    "stillness_threshold": 3,          # 帧间位移小于此值判定为静止（像素）
    "stillness_max_seconds": 1.0,      # 连续静止超过此秒数判定为坏段
    "body_type_ankle_ratio": 0.3,      # 脚踝可见帧占比超过此值判定为全身
    "confidence_threshold": 0.5,       # 关键点置信度阈值
    "pose_coord_weight": 0.5,          # 姿态评分中坐标匹配率的权重
    "pose_angle_weight": 0.3,          # 姿态评分中角度匹配率的权重
    "pose_gesture_weight": 0.2,        # 姿态评分中手部动作加分的权重
    "hand_bonus_wrist_above_shoulder": 0.1,   # 手腕抬过肩膀 +0.1
    "hand_bonus_wrist_span_gt_shoulder": 0.1, # 双手腕间距超肩宽 +0.1
    "hand_bonus_wrist_movement": 0.1,          # 手腕帧间位移>20px +0.1
    "hand_wrist_movement_threshold": 20,       # 手腕帧间位移阈值（像素）
    # ── 面部检测参数（OpenCV Haar cascade）──
    "face_ear_threshold": 0.6,                # EAR 低于此值判定为闭眼
    "face_ear_closed_frames": 5,               # 连续几帧闭眼才触发剔除（去眨眼）
    "face_head_down_ratio_threshold": 0.15,    # 低头比值阈值
    "face_yaw_threshold": 0.12,                # 斜视/侧脸偏移阈值
    "face_pose_score_threshold": 0.4,          # 舞蹈动作排除阈值
    "face_detection_scale_factor": 1.1,        # Haar cascade 参数
    "face_detection_min_neighbors": 5,         # Haar cascade 参数
    "face_detection_min_face_size": 40,        # 最小人脸尺寸（像素）
    # ── 腰胯晃动检测参数 ──
    "hip_sway_threshold_ratio": 0.03,          # 髋部横向位移 > 3%帧宽判定为晃动
    "hip_sway_shoulder_ratio": 0.5,            # 肩膀位移 < 髋位移的50%判定为"纯扭"
    "hip_sway_min_frames": 3,                  # 连续几帧晃动才触发剔除
    "hip_sway_pose_threshold": 0.4,            # pose_score > 此值不剔除（标准展示）
    "hip_sway_gesture_threshold": 0.15,        # 手部动作加分 > 此值不剔除（舞蹈）
    # ── 脚踝/手腕卡顿检测参数（脚踝优先，脚踝不可见时回退手腕）──
    "stutter_big_threshold_ratio": 0.012,       # 脚踝"在动"阈值（帧高比例）
    "stutter_small_threshold_ratio": 0.004,      # 脚踝"卡住"阈值
    "stutter_min_frames": 1,                     # 最少卡几帧算一次卡顿
    "stutter_max_frames": 4,                     # 最多卡几帧（超过算停止不是卡顿）
}


# ── 工具函数 ──────────────────────────────────────────────────────
def _angle_between(p1, p2, p3):
    """计算三点夹角（度），p2 为顶点"""
    if any(p is None for p in (p1, p2, p3)):
        return None
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _pixel_distance(p1, p2):
    """两点间的欧氏距离（像素）"""
    if p1 is None or p2 is None:
        return None
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _copy_model_to_ascii_path(model_src, prefix="pose_landmarker"):
    """
    将模型文件复制到 ASCII 路径（避免 MediaPipe C++ 后端不识别中文路径）
    使用随机文件名防止多实例竞态条件

    Args:
        model_src: 模型源路径
        prefix: 文件名前缀，默认 pose_landmarker

    Returns:
        str: ASCII 目标路径
    """
    import uuid
    ascii_dir = os.path.join(tempfile.gettempdir(), "hermes_mediapipe_models")
    os.makedirs(ascii_dir, exist_ok=True)
    # 使用 UUID 避免多进程/多实例同时运行时的文件冲突
    unique_name = f"{prefix}_{uuid.uuid4().hex[:8]}.task"
    dst = os.path.join(ascii_dir, unique_name)
    if not os.path.exists(dst) and os.path.exists(model_src):
        shutil.copy2(model_src, dst)
    elif os.path.exists(model_src) and os.path.exists(dst) and os.path.getmtime(model_src) > os.path.getmtime(dst):
        shutil.copy2(model_src, dst)
    return dst


# ══════════════════════════════════════════════════════════════════
# ClipVideo Provider
# ══════════════════════════════════════════════════════════════════
class ClipVideo:
    """基于 MediaPipe 骨架检测的智能视频剪辑器"""

    def __init__(self, config=None):
        """
        Args:
            config: 参数字典，覆盖 _DEFAULT_CONFIG 中的默认值
        """
        self.cfg = dict(_DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)

        self._detector = None
        self._face_detector = None
        self._model_path = None

    # ── 公开接口 ──────────────────────────────────────────────

    def compose(self, image_paths, text="", output_path="output.mp4", **kwargs):
        """clip 只处理视频剪辑，不支持图片合成"""
        raise NotImplementedError(
            "ClipVideo 只处理视频剪辑，请使用 moviepy / ffmpeg provider 进行图片合成"
        )

    def clip_video(self, input_path, output_path, target_duration=5.0, target_bbox_size=None):
        """
        核心功能：根据骨架检测自动剪辑视频

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            target_duration: 目标保留时长（秒），默认5秒
            target_bbox_size: 目标骨架bbox面积参考值。为None时自动计算整个视频的平均值

        Returns:
            tuple[str, float]: (输出视频路径, 平均骨架bbox面积)
        """
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe 未安装，请运行: pip install mediapipe"
            )

        print(f"[ClipVideo] 分析视频: {input_path}")
        print(f"[ClipVideo] 目标时长: {target_duration}s")

        # Phase 0: 初始化检测器
        self._init_detector()

        # Phase 1: 扫描视频，逐帧检测骨架 + 场景切换检测
        frame_data = self._scan_video(input_path)

        if not frame_data:
            raise ValueError("未能从视频中提取任何帧数据")

        fps, total_frames = frame_data[0].get("fps", 30), len(frame_data)
        print(f"[ClipVideo] 共扫描 {total_frames} 帧 ({total_frames / fps:.1f}s), "
              f"检测到骨架的帧: {sum(1 for f in frame_data if f['landmarks'] is not None)}")

        # 场景切换检测
        scene_cuts = self._find_scenes(frame_data, fps)
        print(f"[ClipVideo] 场景切换点: {len(scene_cuts)} 个场景")
        for s, e in scene_cuts:
            print(f"  场景: 帧 {s}-{e} ({(e-s)/fps:.1f}s)")

        # Phase 2: 判断全身/半身
        frame_h = frame_data[0].get("frame_h", 720)
        body_type = self._determine_body_type(frame_data, frame_h)
        print(f"[ClipVideo] 身体类型: {'全身 (full)' if body_type == 'full' else '半身 (half)'}")

        # Phase 3: 逐帧评估（剔除规则）
        good_frames = self._evaluate_all_frames(frame_data, fps, body_type)
        good_count = sum(good_frames)
        print(f"[ClipVideo] 好帧: {good_count}/{len(good_frames)} "
              f"({good_count / len(good_frames) * 100:.1f}%)")

        # 面部检测统计
        processed_frames = [f for f in frame_data if f['landmarks'] is not None]
        face_count = sum(1 for f in processed_frames if f.get('face_detected', False))
        print(f"[ClipVideo] 面部检测: 检测到人脸 {face_count}/{len(processed_frames)} 帧")
        eyes_closed_total = sum(1 for f in frame_data if f.get('ear_value') is not None and f['ear_value'] < 0.6)
        head_down_total = sum(1 for f in frame_data if f.get('head_down_ratio') is not None and f['head_down_ratio'] > 0.15)
        print(f"[ClipVideo] 闭眼剔除: {eyes_closed_total} 帧")
        print(f"[ClipVideo] 低头剔除: {head_down_total} 帧")
        hip_sway_total = getattr(self, '_hip_sway_total', 0)
        hip_sway_excluded = getattr(self, '_hip_sway_excluded_dance', 0)
        print(f"[ClipVideo] 腰胯晃动剔除: {hip_sway_total} 帧")
        print(f"[ClipVideo] 其中排除舞蹈动作: {hip_sway_excluded} 帧")
        stutter_total = getattr(self, '_stutter_total', 0)
        print(f"[ClipVideo] 卡顿剔除: {stutter_total} 帧")

        # Phase 3.5: 计算保留规则（姿态1评分 + 手部动作加分）
        pose_scores = self._compute_pose_scores(frame_data, fps)

        # 自动计算整个视频的平均bbox作为骨架大小参考值
        if target_bbox_size is None:
            areas = []
            for fd_entry in frame_data:
                area = self._get_frame_bbox_area(fd_entry)
                if area > 0:
                    areas.append(area)
            target_bbox_size = sum(areas) / len(areas) if areas else 0
            if target_bbox_size > 0:
                print(f"[ClipVideo] 自动计算参考骨架大小: {target_bbox_size:.0f}")

        # Phase 4: 找"保留规则命中率最高"的连贯段（不跨越场景切换点）
        segment = self._find_best_segment(
            good_frames, frame_data, fps, target_duration, pose_scores, scene_cuts,
            target_bbox_size=target_bbox_size
        )
        if segment is None:
            print("[ClipVideo] ⚠️ 未找到足够好的片段，输出原始视频")
            shutil.copy2(input_path, output_path)
            return output_path

        start_frame, end_frame = segment
        actual_duration = (end_frame - start_frame) / fps
        print(f"[ClipVideo] 最佳片段: 帧 {start_frame}-{end_frame} "
              f"({actual_duration:.1f}s)")

        # Phase 5: 输出视频
        self._output_segment(input_path, output_path, start_frame, end_frame, fps)

        print(f"[ClipVideo] ✅ 完成: {output_path}")

        # 在返回前获取平均骨架大小
        avg_size = self.get_average_skeleton_size(frame_data)
        return output_path, avg_size

    def extract_good_frames(self, input_path, max_frames=20, consistent_pose=False):
        """提取视频中姿态评分最高的 N 帧作为图片

        从素材视频中通过骨架扫描提取好帧并保存为JPEG图片，
        用于漫威风格混剪开头中的左侧图片超快切。

        Args:
            input_path: 输入视频路径
            max_frames: 最多提取多少帧（默认20）

        Returns:
            list[str]: 图片路径列表，按姿态评分降序排列
        """
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe 未安装，请运行: pip install mediapipe"
            )

        print(f"[ClipVideo.extract_good_frames] 分析视频: {input_path}")

        # Phase 0: 初始化检测器
        self._init_detector()

        # Phase 1: 扫描视频，逐帧检测骨架
        frame_data = self._scan_video(input_path)

        if not frame_data:
            raise ValueError("未能从视频中提取任何帧数据")

        fps = frame_data[0].get("fps", 30)
        total_frames = len(frame_data)
        detected_frames = [f for f in frame_data if f['landmarks'] is not None]
        print(f"[ClipVideo.extract_good_frames] 共扫描 {total_frames} 帧, "
              f"有骨架帧: {len(detected_frames)}")

        # Phase 2: 计算姿态评分
        pose_scores = self._compute_pose_scores(frame_data, fps)

        # Phase 3: 筛选有骨架且评分高的帧
        scored_indices = []
        for i, f in enumerate(frame_data):
            if f['landmarks'] is not None and f.get('keypoint_positions'):
                scored_indices.append((i, pose_scores[i]))

        # 按评分降序排列
        scored_indices.sort(key=lambda x: -x[1])

        # 取评分最高的max_frames帧
        if not consistent_pose:
            top_indices = [idx for idx, score in scored_indices[:max_frames]]
        else:
            pool = scored_indices[:min(100, len(scored_indices))]
            def _get_pose_vector(fd, idx):
                f = fd[idx]
                kp = f.get('keypoint_positions', {})
                stable_ids = [0, 11, 12, 23, 24, 13, 14, 15, 16]
                vec = []
                for sid in stable_ids:
                    if sid in kp:
                        vec.extend([kp[sid][0], kp[sid][1]])
                    else:
                        vec.extend([0, 0])
                return np.array(vec)
            def _get_bbox_area(fd, idx):
                """计算骨架bbox面积（肩宽×躯干高）"""
                f = fd[idx]
                kp = f.get('keypoint_positions', {})
                if 11 in kp and 12 in kp and 23 in kp and 24 in kp:
                    shoulder_w = abs(kp[12][0] - kp[11][0])
                    hip_w = abs(kp[24][0] - kp[23][0])
                    torso_h = abs((kp[11][1] + kp[12][1])/2 - (kp[23][1] + kp[24][1])/2)
                    return shoulder_w * torso_h
                return 0
            ref_idx = pool[0][0]
            ref_vec = _get_pose_vector(frame_data, ref_idx)
            ref_area = _get_bbox_area(frame_data, ref_idx)
            # 如果参考帧没有完整骨架（area=0），放宽大小限制
            skip_area_check = (ref_area == 0)
            if skip_area_check:
                print(f"[ClipVideo] 参考帧 #{ref_idx} 无完整骨架，跳过大小过滤")
            else:
                print(f"[ClipVideo] 参考帧 #{ref_idx} 骨架面积={ref_area:.0f}")
            scored_similar = []
            for idx, score in pool:
                vec = _get_pose_vector(frame_data, idx)
                dist = np.linalg.norm(vec - ref_vec)
                combined = score - dist * 0.05
                # 只在两帧都有完整骨架时才加面积惩罚
                if not skip_area_check:
                    area = _get_bbox_area(frame_data, idx)
                    if area > 0:
                        area_penalty = abs(area - ref_area) * 0.002
                        combined -= area_penalty
                scored_similar.append((idx, combined))
            scored_similar.sort(key=lambda x: -x[1])
            top_indices = [idx for idx, s in scored_similar[:max_frames]]
            print(f"[ClipVideo] 姿态一致性模式: 参考帧 #{ref_idx}, 选 {len(top_indices)} 帧")

        if not top_indices:
            print("[ClipVideo.extract_good_frames] ⚠️ 无好帧，从视频均匀采样")
            # 回退：从视频中均匀采样
            total = len(frame_data)
            step = max(1, total // max_frames)
            top_indices = list(range(0, total, step))[:max_frames]

        print(f"[ClipVideo.extract_good_frames] 选取 {len(top_indices)} 帧")

        # Phase 4: 用 OpenCV 提取这些帧并保存为 JPEG
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {input_path}")

        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "output", "temp_frames"
        ) if '__file__' in dir() else "output/temp_frames"
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        image_paths = []
        target_set = set(top_indices)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in target_set:
                    out_path = os.path.join(output_dir, f"good_frame_{frame_idx:06d}.jpg")
                    # cv2.imwrite 在中文路径下可能失败，改用 cv2.imencode + 字节写入
                    success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if success:
                        with open(out_path, 'wb') as f:
                            f.write(buf.tobytes())
                    if not os.path.exists(out_path):
                        print(f"  [⚠️] extract_good_frames: 未能保存文件(中文路径?): {out_path}")
                    image_paths.append(out_path)
                frame_idx += 1
        finally:
            cap.release()

        print(f"[ClipVideo.extract_good_frames] ✅ 提取 {len(image_paths)} 张图片到 {output_dir}")
        return image_paths

    def get_average_skeleton_size(self, frame_data):
        """计算该段视频的平均骨架大小（bbox面积）

        Args:
            frame_data: _scan_video 的返回值

        Returns:
            float: 平均骨架面积，若无骨架数据则返回 0
        """
        areas = [f["bbox_area"] for f in frame_data if f.get("bbox_area") is not None]
        if not areas:
            return 0
        return sum(areas) / len(areas)

    def batch_clip(self, video_paths, output_path, total_duration=15.0):
        """
        批量处理：多个视频 → 分别分析 → 各自取一段 → 拼接成总长

        Args:
            video_paths: 输入视频路径列表
            output_path: 输出视频路径
            total_duration: 最终总时长（秒），默认15秒

        Returns:
            str: 输出视频路径
        """
        if not video_paths:
            raise ValueError("视频路径列表为空")

        per_duration = total_duration / len(video_paths)
        temp_dir = tempfile.mkdtemp(prefix="clip_batch_")
        temp_clips = []

        try:
            for i, vpath in enumerate(video_paths):
                print(f"\n{'=' * 50}")
                print(f"[Batch] 处理第 {i+1}/{len(video_paths)} 个视频: {vpath}")
                temp_out = os.path.join(temp_dir, f"clip_{i:03d}.mp4")
                self.clip_video(vpath, temp_out, target_duration=per_duration)
                temp_clips.append(temp_out)

            # 用 OpenCV 拼接
            print(f"\n[Batch] 拼接 {len(temp_clips)} 个片段...")
            self._concat_videos(temp_clips, output_path)
            print(f"[Batch] ✅ 完成: {output_path}")

        finally:
            # 清理临时文件
            for f in temp_clips:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

        return output_path

    # ── 初始化 ────────────────────────────────────────────────

    def _init_detector(self):
        """初始化 MediaPipe PoseLandmarker（模型路径用 ASCII）"""
        if self._detector is not None:
            return

        # 查找模型文件
        search_paths = [
            # 当前目录下的 _models
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_models",
                "pose_landmarker_lite.task",
            ),
            # 上一级 providers 目录下的 _models
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "video",
                "_models",
                "pose_landmarker_lite.task",
            ),
            # 用户目录（之前可能已复制）
            os.path.join(os.path.expanduser("~"), "pose_landmarker_lite.task"),
        ]

        model_src = None
        for p in search_paths:
            if os.path.exists(p):
                model_src = p
                break

        if model_src is None:
            raise FileNotFoundError(
                f"未找到 pose_landmarker_lite.task，搜索路径: {search_paths}\n"
                "请从 https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/"
                "pose_landmarker_lite.task 下载并放入 providers/video/_models/"
            )

        # 复制到 ASCII 路径
        ascii_path = _copy_model_to_ascii_path(model_src)
        self._model_path = ascii_path

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ascii_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)
        print(f"[ClipVideo] ✅ 骨架检测器已加载 (模型: {os.path.basename(ascii_path)})")

    def _detect_face_and_eyes(self, bgr_frame, keypoints, frame_h):
        """
        使用 OpenCV Haar cascade 检测面部和眼睛

        Args:
            bgr_frame: BGR 图像帧
            keypoints: Pose 骨架关键点 (用于辅助定位人脸区域)
            frame_h: 帧高度

        Returns:
            face_rect: (x, y, w, h) 人脸框或 None
            ear_value: float (0.0~2.0, 眼框宽高比，越小越闭)
            head_down_ratio: float 或 None
        """
        # 懒初始化 Haar cascade
        if not hasattr(self, '_haar_face_cascade') or self._haar_face_cascade is None:
            try:
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path):
                    print("[ClipVideo] ⚠️ OpenCV Haar cascade 文件未找到，面部规则将跳过")
                    self._haar_face_cascade = "disabled"
                    self._haar_eye_cascade = "disabled"
                else:
                    self._haar_face_cascade = cv2.CascadeClassifier(face_cascade_path)
                    self._haar_eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                    if self._haar_face_cascade.empty() or self._haar_eye_cascade.empty():
                        print("[ClipVideo] ⚠️ OpenCV Haar cascade 加载失败，面部规则将跳过")
                        self._haar_face_cascade = "disabled"
                        self._haar_eye_cascade = "disabled"
            except Exception as e:
                print(f"[ClipVideo] ⚠️ Haar cascade 加载失败: {e}，面部规则将跳过")
                self._haar_face_cascade = "disabled"
                self._haar_eye_cascade = "disabled"

        if self._haar_face_cascade == "disabled":
            return None, None, None

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        h, w = bgr_frame.shape[:2]

        scale_factor = self.cfg.get("face_detection_scale_factor", 1.1)
        min_neighbors = self.cfg.get("face_detection_min_neighbors", 5)
        min_face_size = self.cfg.get("face_detection_min_face_size", 40)

        faces = self._haar_face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face_size, min_face_size),
        )

        face_rect = None
        face_roi = None

        if len(faces) > 0:
            # 取面积最大的人脸（确保是主体人物）
            best_face = max(faces, key=lambda r: r[2] * r[3])
            fx, fy, fw, fh = best_face
            face_rect = (fx, fy, fw, fh)
            face_roi = gray[fy:fy+fh, fx:fx+fw]
        else:
            # 没检测到人脸，用骨架关键点辅助估计人脸区域（基于鼻子位置）
            if keypoints and len(keypoints) > 0:
                nose_idx = 0  # LANDMARK_NOSE
                if nose_idx < len(keypoints) and keypoints[nose_idx] is not None:
                    nx, ny = keypoints[nose_idx]
                    # 估算人脸区域：鼻子周围约 80x80 区域
                    face_size = int(frame_h * 0.11)  # ~11% 帧高
                    fx = max(0, nx - face_size // 2)
                    fy = max(0, ny - int(face_size * 0.3))  # 鼻子偏下，向上取更多
                    fw = min(face_size, w - fx)
                    fh = min(int(face_size * 1.2), h - fy)  # 脸比宽稍高
                    if fw > 10 and fh > 10:
                        face_rect = (fx, fy, fw, fh)
                        face_roi = gray[fy:fy+fh, fx:fx+fw]

        # 计算 EAR（眼睛宽高比）
        ear_value = None
        if face_roi is not None:
            eyes = self._haar_eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.15,
                minNeighbors=3,
                minSize=(10, 10),
            )
            if len(eyes) > 0:
                aspects = []
                for (ex, ey, ew, eh) in eyes:
                    if eh > 0:
                        aspect = ew / eh
                        # 过滤明显不合理的 aspect 值
                        if 0.2 < aspect < 4.0:
                            aspects.append(aspect)
                if aspects:
                    ear_value = sum(aspects) / len(aspects)
                    # 限制在合理范围
                    ear_value = max(0.05, min(2.0, ear_value))
            else:
                # 只有人脸足够大（>80px高）且检测不到眼睛才判定为闭眼
                # 小脸/远距离场景下眼睛太小，Haar检测不到是正常的
                if face_rect and face_rect[3] > 80:
                    ear_value = 0.05
                # 脸太小 -> 留 ear_value=None，跳过闭眼检测

        # 计算低头比值
        head_down_ratio = None
        if face_rect is not None and keypoints and len(keypoints) > 0:
            nose_idx = 0  # LANDMARK_NOSE
            if nose_idx < len(keypoints) and keypoints[nose_idx] is not None:
                nose_y = keypoints[nose_idx][1]
                # 肩膀中间 y 坐标
                shoulder_l_y = keypoints[11][1] if len(keypoints) > 11 and keypoints[11] is not None else None
                shoulder_r_y = keypoints[12][1] if len(keypoints) > 12 and keypoints[12] is not None else None
                if shoulder_l_y is not None and shoulder_r_y is not None:
                    shoulder_mid_y = (shoulder_l_y + shoulder_r_y) / 2.0
                    head_down_ratio = (nose_y - shoulder_mid_y) / frame_h
                elif face_rect is not None:
                    # 没有肩膀数据时，用人脸框在画面中的相对位置
                    face_center_y = face_rect[1] + face_rect[3] / 2
                    head_down_ratio = (face_center_y - frame_h * 0.35) / frame_h

        return face_rect, ear_value, head_down_ratio

    def _check_eyes_closed(self, ear_value, closed_eye_counter):
        """
        闭眼检测规则：区分眨眼 vs 闭眼（使用 Haar cascade 的 EAR 宽高比）

        Args:
            ear_value: 当前帧的 EAR 值（Haar 眼睛框宽高比，检测不到眼≈0.05）
            closed_eye_counter: 当前连续闭眼帧数

        Returns:
            rating: "eyes_closed" | None
            updated_counter: 更新后的闭眼计数器
        """
        ear_threshold = self.cfg["face_ear_threshold"]
        closed_threshold = self.cfg["face_ear_closed_frames"]

        if ear_value < ear_threshold:
            closed_eye_counter += 1
        else:
            closed_eye_counter = 0

        if closed_eye_counter >= closed_threshold:
            return "eyes_closed", closed_eye_counter

        return None, closed_eye_counter

    def _check_head_down(self, head_down_ratio, keypoints, frame_h, current_pose_score):
        """
        低头检测规则：低头退步时剔除，斜视和舞蹈动作不算

        Args:
            head_down_ratio: 低头比值 (nose_y - shoulder_mid_y) / frame_h
            keypoints: Pose 骨架关键点像素坐标列表
            frame_h: 帧高度
            current_pose_score: 当前帧的姿态1评分 (0.0~1.0)

        Returns:
            rating: "head_down" | None
        """
        if head_down_ratio is None:
            return None

        ratio_threshold = self.cfg["face_head_down_ratio_threshold"]
        yaw_threshold = self.cfg["face_yaw_threshold"]
        pose_score_threshold = self.cfg["face_pose_score_threshold"]

        # 检查斜视/侧脸：鼻子 x 坐标相对肩膀中心的偏移
        if keypoints and len(keypoints) > 12:
            nose_x = keypoints[0][0] if keypoints[0] is not None else None
            shoulder_l_x = keypoints[11][0] if keypoints[11] is not None else None
            shoulder_r_x = keypoints[12][0] if keypoints[12] is not None else None
            if all(x is not None for x in [nose_x, shoulder_l_x, shoulder_r_x]):
                shoulder_mid_x = (shoulder_l_x + shoulder_r_x) / 2.0
                shoulder_width = abs(shoulder_r_x - shoulder_l_x)
                if shoulder_width > 0:
                    yaw_ratio = abs(nose_x - shoulder_mid_x) / shoulder_width
                    # 斜视/侧脸不算低头
                    if yaw_ratio > yaw_threshold:
                        return None

        # 舞蹈动作不算（姿态评分高说明是正常舞蹈姿态）
        if current_pose_score > pose_score_threshold:
            return None

        if head_down_ratio > ratio_threshold:
            return "head_down"

        return None

    def _check_hip_sway(self, keypoints, prev_keypoints, frame_w, current_pose_score,
                        gesture_bonus, sway_counter):
        """
        Rule 6 - 腰胯过度晃动检测：
        用肩髋同/异步判定晃动类型。只有"腰扭上身不动"的帧才会被剔除，
        不误杀舞蹈动作和转圈展示。

        核心逻辑：
        - 髋部中点横向位移 > 3%帧宽 = 明显晃动
        - 但肩膀没跟着动（位移 < 髋位移的50%）= 纯腰胯扭动
        - 连续性要求：连续 sway_counter >= 3 帧才触发剔除

        Args:
            keypoints: 当前帧关键点像素坐标列表
            prev_keypoints: 上一帧关键点像素坐标列表
            frame_w: 帧宽度（像素）
            current_pose_score: 当前帧的姿态1评分 (0.0~1.0)
            gesture_bonus: 当前帧的手部动作加分 (0.0~0.3)
            sway_counter: 当前连续晃动帧数

        Returns:
            tuple: (rating: "hip_sway" | None, sway_counter: int, excluded_by_safety: bool)
                   excluded_by_safety: True 表示本应判定为晃动但被安全条件排除（舞蹈/展示/转圈）
        """
        threshold_ratio = self.cfg["hip_sway_threshold_ratio"]
        shoulder_ratio = self.cfg["hip_sway_shoulder_ratio"]
        min_frames = self.cfg["hip_sway_min_frames"]
        pose_threshold = self.cfg["hip_sway_pose_threshold"]
        gesture_threshold = self.cfg["hip_sway_gesture_threshold"]

        # 需要当前帧和上一帧都有足够的关键点数据
        if not keypoints or not prev_keypoints:
            return None, 0, False
        if len(keypoints) <= max(LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_SHOULDER):
            return None, 0, False
        if len(prev_keypoints) <= max(LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_SHOULDER):
            return None, 0, False

        # 1. 计算髋部中点横向位移
        left_hip_curr = keypoints[LANDMARK_LEFT_HIP]
        right_hip_curr = keypoints[LANDMARK_RIGHT_HIP]
        left_hip_prev = prev_keypoints[LANDMARK_LEFT_HIP]
        right_hip_prev = prev_keypoints[LANDMARK_RIGHT_HIP]

        if any(p is None for p in [left_hip_curr, right_hip_curr, left_hip_prev, right_hip_prev]):
            return None, 0, False

        hip_mid_x_curr = (left_hip_curr[0] + right_hip_curr[0]) / 2.0
        hip_mid_x_prev = (left_hip_prev[0] + right_hip_prev[0]) / 2.0
        hip_dx = abs(hip_mid_x_curr - hip_mid_x_prev)

        # 2. 计算肩部中点横向位移（对照）
        left_shoulder_curr = keypoints[LANDMARK_LEFT_SHOULDER]
        right_shoulder_curr = keypoints[LANDMARK_RIGHT_SHOULDER]
        left_shoulder_prev = prev_keypoints[LANDMARK_LEFT_SHOULDER]
        right_shoulder_prev = prev_keypoints[LANDMARK_RIGHT_SHOULDER]

        if any(p is None for p in [left_shoulder_curr, right_shoulder_curr,
                                    left_shoulder_prev, right_shoulder_prev]):
            return None, 0, False

        shoulder_mid_x_curr = (left_shoulder_curr[0] + right_shoulder_curr[0]) / 2.0
        shoulder_mid_x_prev = (left_shoulder_prev[0] + right_shoulder_prev[0]) / 2.0
        shoulder_dx = abs(shoulder_mid_x_curr - shoulder_mid_x_prev)

        # 3. 判定是否为"纯腰胯扭动"
        is_pure_hip_sway = False
        excluded_by_safety = False
        if hip_dx > frame_w * threshold_ratio:
            # 髋部有明显左右晃动
            if shoulder_dx < hip_dx * shoulder_ratio:
                # 肩膀没跟着动 → 只有腰在扭、上身僵着
                is_pure_hip_sway = True

        # 4. 排除条件（这些情况不算过度晃动）
        if is_pure_hip_sway:
            # 标准展示姿态 → 不剔除
            if current_pose_score > pose_threshold:
                is_pure_hip_sway = False
                excluded_by_safety = True
            # 手在动/抬手/张开 → 舞蹈动作 → 不剔除
            elif gesture_bonus > gesture_threshold:
                is_pure_hip_sway = False
                excluded_by_safety = True
            # 肩膀跟着动 → 全身协调动作/转圈 → 不剔除
            elif shoulder_dx >= hip_dx * shoulder_ratio:
                is_pure_hip_sway = False
                excluded_by_safety = True

        # 5. 连续性要求
        if is_pure_hip_sway:
            sway_counter += 1
        else:
            sway_counter = 0

        if sway_counter >= min_frames:
            return "hip_sway", sway_counter, excluded_by_safety

        return None, sway_counter, excluded_by_safety

    def _check_stutter(self, keypoints, prev_keypoints, prev2_keypoints, frame_h, stutter_counter):
        """
        Rule 7 - 四肢卡顿检测：
        检测"移动→卡住→再移动"的模式（stutter = 动→卡→动），用3帧窗口判断。
        只关注四肢末端（左腕15、右腕16、左踝27、右踝28），忽略躯干点。

        核心逻辑：
        - 如果上一帧在动（prev2→prev 位移大）而当前帧卡住（prev→curr 位移小）
          → 进入卡顿状态，stutter_counter += 1
        - 如果之前卡住过（stutter_counter > 0）且当前帧又开始动
          → 这就是一次完整的卡顿事件，返回 "stutter"
        - 如果卡住后一直没恢复（stutter_counter >= max_frames）→ 是"停止"不是"卡顿"

        Args:
            keypoints: 当前帧关键点像素坐标列表
            prev_keypoints: 上一帧关键点像素坐标列表
            prev2_keypoints: 上上帧关键点像素坐标列表
            frame_h: 帧高度（像素）
            stutter_counter: 当前连续卡住帧数

        Returns:
            tuple: (rating: "stutter" | None, stutter_counter: int)
        """
        # 策略：脚踝优先，脚踝不可见时回退到手腕
        ANKLE_IDXS = [27, 28]
        WRIST_IDXS = [15, 16]
        
        max_frames = self.cfg["stutter_max_frames"]
        min_frames = self.cfg["stutter_min_frames"]
        
        # 检查脚踝在当前三帧中是否都有有效数据
        def _has_ankle_data(kp_list):
            """检查脚踝关键点(27,28)在帧列表中是否都有 x/y 数据"""
            for kp in kp_list:
                if kp is None:
                    return False
                for idx in ANKLE_IDXS:
                    if idx >= len(kp) or kp[idx] is None:
                        return False
            return True
        
        three_frames = [keypoints, prev_keypoints, prev2_keypoints]
        ankles_ok = _has_ankle_data(three_frames)
        
        if ankles_ok:
            # 脚踝有数据 → 脚踝+手腕同时检测，脚踝阈值灵敏
            LIMB_INDICES = ANKLE_IDXS + WRIST_IDXS
            big_thresh = frame_h * self.cfg["stutter_big_threshold_ratio"]
            small_thresh = frame_h * self.cfg["stutter_small_threshold_ratio"]
        else:
            # 脚踝不可见（半身镜头） → 只盯手腕，阈值更不灵敏少误判
            LIMB_INDICES = WRIST_IDXS
            big_thresh = frame_h * 0.025
            small_thresh = frame_h * 0.008

        if not all([keypoints, prev_keypoints, prev2_keypoints]):
            return None, 0
        if any(len(kp) <= max(LIMB_INDICES) for kp in [keypoints, prev_keypoints, prev2_keypoints]):
            return None, 0

        for idx in LIMB_INDICES:
            d1 = _pixel_distance(keypoints[idx], prev_keypoints[idx])
            d2 = _pixel_distance(prev_keypoints[idx], prev2_keypoints[idx])

            if d1 is None or d2 is None:
                continue

            if d2 > big_thresh and d1 < small_thresh:
                # 刚才在动但现在卡住了 → 进入卡顿状态
                stutter_counter += 1
                continue

            if stutter_counter >= min_frames and d1 > big_thresh:
                # 之前卡住过（达到最少卡顿帧数），现在又开始动了 → 这就是一次卡顿
                stutter_counter = 0
                return "stutter", stutter_counter

            if d1 > big_thresh:
                stutter_counter = 0

        # 如果持续卡住没恢复（stop模式），不是stutter
        if stutter_counter >= max_frames:
            stutter_counter = 0  # 卡太久了说明是停止不是卡顿

        return None, stutter_counter

    # ── Phase 1: 扫描视频 ────────────────────────────────────

    def _scan_video(self, input_path):
        """
        逐帧扫描视频，骨架检测（每隔1帧处理1帧加速）

        Returns:
            list[dict]: 每帧数据
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Phase 1] 视频: {frame_w}x{frame_h}, {fps:.2f}fps, "
              f"{total_frames}帧 ({total_frames / fps:.1f}s)")

        frame_data = []
        frame_idx = 0
        last_progress = -1
        # 跳帧处理：每 frame_skip 帧只处理1帧，减少CPU占用
        frame_skip = 2  # 处理1/3的帧 → ~105FPS等效
        frame_counter = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                # 跳帧：不检测骨架，只记录帧信息
                process_this = (frame_counter % (frame_skip + 1) == 0)

                frame_info = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else 0,
                    "landmarks": None,
                    "num_landmarks": 0,
                    "keypoint_positions": [],
                    "fps": fps,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "bbox_area": None,
                    "face_detected": False,
                    "ear_value": None,
                    "head_down_ratio": None,
                }

                if process_this:
                    landmarks_norm, keypoints_pixel = self._detect_frame(frame)
                    frame_info["landmarks"] = landmarks_norm
                    frame_info["num_landmarks"] = len(keypoints_pixel)
                    frame_info["keypoint_positions"] = keypoints_pixel
                    # 计算骨架 bbox 面积（归一化坐标 * 帧宽高 = 像素坐标）
                    if landmarks_norm:
                        h, w = frame.shape[:2]
                        xs = [lm.x * w for lm in landmarks_norm]
                        ys = [lm.y * h for lm in landmarks_norm]
                        bbox_w = max(xs) - min(xs)
                        bbox_h = max(ys) - min(ys)
                        frame_info["bbox_area"] = bbox_w * bbox_h

                        # 面部检测（Haar cascade）：只在有骨架检测结果的帧上运行
                        face_rect, ear_val, hd_ratio = self._detect_face_and_eyes(frame, keypoints_pixel, frame_h)
                        if face_rect is not None:
                            frame_info["face_detected"] = True
                        if ear_val is not None:
                            frame_info["ear_value"] = ear_val
                        if hd_ratio is not None:
                            frame_info["head_down_ratio"] = hd_ratio

                frame_data.append(frame_info)
                frame_idx += 1

                # 进度显示
                if total_frames > 0 and frame_counter % 30 == 0:
                    progress = int(frame_counter / total_frames * 100) if total_frames > 0 else 0
                    print(f"  [进度] {progress}% ({frame_counter}/{total_frames}帧)")
        finally:
            cap.release()
        print(f"  [完成] 扫描结束，共 {len(frame_data)} 帧")
        return frame_data

    def _detect_frame(self, bgr_frame):
        """
        对单帧进行骨架检测

        Args:
            bgr_frame: OpenCV BGR 帧

        Returns:
            landmarks_norm: list of NormalizedLandmark (33个) 或 None
            keypoints_pixel: list of (x, y) 像素坐标
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            return None, []

        landmarks = result.pose_landmarks[0]  # 取第一个人
        h, w = bgr_frame.shape[:2]
        keypoints = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            keypoints.append((px, py))

        return landmarks, keypoints

    # ── 场景切换检测 ─────────────────────────────────────────

    def _find_scenes(self, frame_data, fps):
        """
        检测视频中的场景切换点。

        场景切换判定：
        1. 连续5帧以上检测不到骨架（置信度<0.5）-> 场景结束
        2. 连续5帧以上检测到骨架，且位置与上一场景偏差大 -> 新场景开始
        3. 只要骨架保持连续有检测，就是同一个场景

        输出: [(start_frame, end_frame), ...] 场景列表
        """
        n = len(frame_data)
        if n == 0:
            return [(0, 0)]

        SCENE_GAP_FRAMES = 5  # 连续5帧无骨架 → 场景结束

        # Step 1: 找到所有"无人区间"（连续无骨架帧段）
        no_skeleton_runs = []  # [(start, end), ...] 无骨架区间
        run_start = None
        for i, f in enumerate(frame_data):
            has_skeleton = f["landmarks"] is not None
            if not has_skeleton:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_length = i - run_start
                    if run_length >= SCENE_GAP_FRAMES:
                        no_skeleton_runs.append((run_start, i - 1))
                    run_start = None
        # 末尾
        if run_start is not None:
            run_length = n - run_start
            if run_length >= SCENE_GAP_FRAMES:
                no_skeleton_runs.append((run_start, n - 1))

        # Step 2: 用无人区间切分场景
        scenes = []
        last_end = -1
        for gap_start, gap_end in no_skeleton_runs:
            if gap_start > last_end + 1:
                scenes.append((last_end + 1, gap_start - 1))
            last_end = gap_end

        # 最后一段
        if last_end < n - 1:
            scenes.append((last_end + 1, n - 1))

        # Step 3: 检查每个场景内部是否有位置偏差大的子段（场景内镜头切换）
        # 对每个场景，检测骨架位置的突变（鼻子水平位移 > 50%画面高度）
        refined_scenes = []
        for scene_start, scene_end in scenes:
            # 找这个场景中有检测结果的帧
            detected_indices = []
            for i in range(scene_start, scene_end + 1):
                if frame_data[i]["landmarks"] is not None:
                    detected_indices.append(i)

            if len(detected_indices) < SCENE_GAP_FRAMES:
                # 场景太短 → 直接保留
                refined_scenes.append((scene_start, scene_end))
                continue

            # 找突变点：鼻子位移超过50%画面高度
            sub_cuts = []
            prev_nose = None
            prev_idx = None
            for idx in detected_indices:
                kp = frame_data[idx].get("keypoint_positions", [])
                if len(kp) > LANDMARK_NOSE:
                    curr_nose = kp[LANDMARK_NOSE]
                    if prev_nose is not None and prev_idx is not None:
                        d = _pixel_distance(curr_nose, prev_nose)
                        frame_h = frame_data[idx].get("frame_h", 720)
                        if d is not None and d > frame_h * 0.5:
                            # 位置突变 → 场景内切换
                            sub_cuts.append(idx)
                    prev_nose = curr_nose
                    prev_idx = idx

            if not sub_cuts:
                refined_scenes.append((scene_start, scene_end))
            else:
                # 按突变点切分子场景
                sub_start = scene_start
                for cut in sorted(set(sub_cuts)):
                    if cut > sub_start:
                        refined_scenes.append((sub_start, cut - 1))
                    sub_start = cut
                if sub_start <= scene_end:
                    refined_scenes.append((sub_start, scene_end))

        # Step 4: 合并太短的场景（<0.5秒的合并到前一个）
        min_scene_len = int(fps * 0.5)
        merged = []
        for s, e in refined_scenes:
            if s >= e:
                continue
            if merged and (e - s) < min_scene_len:
                # 合并到前一个场景
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        if not merged:
            merged.append((0, n - 1))

        return merged

    # ── Phase 2: 判断全身/半身 ──────────────────────────────

    def _determine_body_type(self, all_frame_data, frame_h):
        """
        判断是全身（full）还是半身（half）

        统计有检测的帧中，脚踝可见的占比。
        > 30% → full，否则 half
        """
        ankle_ratio = self.cfg["body_type_ankle_ratio"]
        detected_frames = [f for f in all_frame_data if f["landmarks"] is not None]

        if not detected_frames:
            return "half"  # 没有检测到人，保守默认半身

        ankle_visible_count = 0
        for f in detected_frames:
            lms = f["landmarks"]
            # 左踝(27)或右踝(28) 置信度 > 阈值
            left_ankle_ok = (
                lms[LANDMARK_LEFT_ANKLE].visibility
                >= self.cfg["confidence_threshold"]
            )
            right_ankle_ok = (
                lms[LANDMARK_RIGHT_ANKLE].visibility
                >= self.cfg["confidence_threshold"]
            )
            if left_ankle_ok or right_ankle_ok:
                ankle_visible_count += 1

        ratio = ankle_visible_count / len(detected_frames)
        print(f"[Phase 2] 脚踝可见帧: {ankle_visible_count}/{len(detected_frames)} "
              f"({ratio:.1%}, 阈值={ankle_ratio})")
        return "full" if ratio > ankle_ratio else "half"

    # ── Phase 3: 逐帧评估（剔除规则） ─────────────────────────

    def _evaluate_all_frames(self, frame_data, fps, body_type):
        """
        对所有帧逐帧评估，结合5帧滑动窗口

        Returns:
            list[bool]: True=好帧, False=坏帧
        """
        still_counter = 0
        closed_eye_counter = 0
        sway_counter = 0
        stutter_counter = 0
        self._hip_sway_total = 0
        self._hip_sway_excluded_dance = 0
        stillness_max_frames = int(
            self.cfg["stillness_max_seconds"] * fps
        )

        # 找到上一个有检测结果的帧（跳帧时只对有检测结果的帧做评估）
        def _find_prev_detected(idx):
            """从 idx-1 往前找，返回第一个有检测结果的帧数据"""
            for j in range(idx - 1, -1, -1):
                if frame_data[j]["landmarks"] is not None:
                    return frame_data[j]
            return None

        def _find_prev2_detected(idx):
            """从 idx-1 往前找，返回第二个有检测结果的帧数据（用于卡顿检测的 prev2）"""
            found = []
            for j in range(idx - 1, -1, -1):
                if frame_data[j]["landmarks"] is not None:
                    found.append(frame_data[j])
                    if len(found) == 2:
                        return found[1]
            return None

        # 初始评估 — 只评估有检测结果的帧
        raw_ratings = []  # "good" | "folded" | "incomplete" | "still" | "eyes_closed" | "head_down" | "hip_sway" | "stutter"
        prev_kp_for_pose = None
        frame_h = frame_data[0].get("frame_h", 720) if frame_data else 720

        for i, f in enumerate(frame_data):
            if f["landmarks"] is None:
                # 跳过的帧：暂时标记 None，后面用插值补齐
                raw_ratings.append(None)
                continue

            prev_f = _find_prev_detected(i)
            prev_lms = prev_f["landmarks"] if prev_f else None
            prev_kp = prev_f["keypoint_positions"] if prev_f else []

            # 找到上上帧（用于卡顿检测的 prev2）
            prev2_f = _find_prev2_detected(i)
            prev2_kp = prev2_f["keypoint_positions"] if prev2_f else []

            # 计算当前帧的姿态1评分（用于低头检测中的舞蹈动作排除）
            kp = f.get("keypoint_positions", [])
            current_pose_score = self._calc_pose_score(kp, prev_kp_for_pose, frame_h)
            if kp and len(kp) >= LANDMARK_RIGHT_ANKLE + 1:
                prev_kp_for_pose = kp

            rating, still_counter, closed_eye_counter, sway_counter, sway_excluded, stutter_counter = self._evaluate_frame(
                f["landmarks"],
                f["keypoint_positions"],
                prev_lms,
                prev_kp,
                prev2_kp,
                body_type,
                f["frame_h"],
                f["frame_w"],
                still_counter,
                stillness_max_frames,
                closed_eye_counter,
                current_pose_score,
                f.get("face_detected", False),
                f.get("ear_value"),
                f.get("head_down_ratio"),
                sway_counter,
                stutter_counter,
            )
            raw_ratings.append(rating)
            self._hip_sway_excluded_dance += sway_excluded

        # 将跳过的帧的评分用最近的检测帧填补
        # 先正向填补（从前到后）
        last_rating = "incomplete"
        for i in range(len(raw_ratings)):
            if raw_ratings[i] is not None:
                last_rating = raw_ratings[i]
            else:
                raw_ratings[i] = last_rating

        # 5帧滑动窗口平滑
        smoothed = self._sliding_window_smooth(raw_ratings, window_radius=2)

        # 统计面部检测剔除结果
        eyes_closed_count = sum(1 for r in raw_ratings if r == "eyes_closed")
        head_down_count = sum(1 for r in raw_ratings if r == "head_down")
        hip_sway_count = sum(1 for r in raw_ratings if r == "hip_sway")
        stutter_count = sum(1 for r in raw_ratings if r == "stutter")
        self._stutter_total = stutter_count
        self._hip_sway_total = hip_sway_count
        if eyes_closed_count > 0:
            print(f"[Phase 3] 闭眼剔除帧: {eyes_closed_count} 帧")
        if head_down_count > 0:
            print(f"[Phase 3] 低头剔除帧: {head_down_count} 帧")
        if hip_sway_count > 0:
            print(f"[Phase 3] 腰胯过度晃动剔除: {hip_sway_count} 帧")
        if stutter_count > 0:
            print(f"[Phase 3] 卡顿剔除: {stutter_count} 帧")

        return smoothed

    def _evaluate_frame(
        self,
        landmarks,
        keypoints,
        prev_landmarks,
        prev_keypoints,
        prev2_keypoints,
        body_type,
        frame_h,
        frame_w,
        still_counter,
        stillness_max_frames,
        closed_eye_counter=0,
        current_pose_score=0.0,
        face_detected=False,
        ear_value=None,
        head_down_ratio=None,
        sway_counter=0,
        stutter_counter=0,
    ):
        """
        评估一帧的好坏

        Returns:
            rating: "good" | "folded" | "incomplete" | "still" | "eyes_closed" | "head_down" | "hip_sway" | "stutter"
            still_counter: 更新后的静止计数器
            closed_eye_counter: 更新后的闭眼计数器
            sway_counter: 更新后的腰胯晃动计数器
            sway_excluded_count: 本次调用被安全条件排除的晃动帧数（0 或 1）
            stutter_counter: 更新后的卡顿计数器
        """
        # 如果没检测到骨架
        if landmarks is None:
            return "incomplete", still_counter, closed_eye_counter, sway_counter, 0, stutter_counter

        # Rule 2: 不完整检测（先检查，因为需要期望的关键点）
        incomplete = self._check_incomplete(landmarks, body_type)
        if incomplete:
            return "incomplete", still_counter, closed_eye_counter, sway_counter, 0, stutter_counter

        # Rule 1: 折叠检测
        folded = self._check_folded(
            landmarks, keypoints, prev_landmarks, prev_keypoints, frame_h
        )
        if folded:
            return "folded", still_counter, closed_eye_counter, sway_counter, 0, stutter_counter

        # Rule 3: 静止检测
        is_still, still_counter = self._check_still(
            landmarks,
            keypoints,
            prev_landmarks,
            prev_keypoints,
            still_counter,
            stillness_max_frames,
        )
        if is_still:
            return "still", still_counter, closed_eye_counter, sway_counter, 0, stutter_counter

        # Rule 4: 闭眼检测（Haar cascade EAR，需要帧间历史区分眨眼 vs 闭眼）
        if face_detected and ear_value is not None:
            rating, closed_eye_counter = self._check_eyes_closed(ear_value, closed_eye_counter)
            if rating:
                return rating, still_counter, closed_eye_counter, sway_counter, 0, stutter_counter

        # Rule 5: 低头检测（斜视和舞蹈动作不算）
        if face_detected and head_down_ratio is not None:
            rating = self._check_head_down(head_down_ratio, keypoints, frame_h, current_pose_score)
            if rating:
                return rating, still_counter, closed_eye_counter, sway_counter, 0, stutter_counter

        # Rule 6: 腰胯过度晃动（不误杀舞蹈/转圈展示）
        excluded_this_frame = 0
        if keypoints and prev_keypoints:
            gesture_bonus = self._calc_gesture_bonus(keypoints, frame_h, prev_keypoints)
            rating, sway_counter, _excluded = self._check_hip_sway(
                keypoints, prev_keypoints, frame_w, current_pose_score,
                gesture_bonus, sway_counter
            )
            if _excluded:
                excluded_this_frame = 1
            if rating:
                return rating, still_counter, closed_eye_counter, sway_counter, excluded_this_frame, stutter_counter

        # Rule 7: 四肢卡顿检测（动→卡→动模式）
        if keypoints and prev_keypoints and prev2_keypoints:
            rating, stutter_counter = self._check_stutter(
                keypoints, prev_keypoints, prev2_keypoints, frame_h, stutter_counter
            )
            if rating:
                return rating, still_counter, closed_eye_counter, sway_counter, excluded_this_frame, stutter_counter

        return "good", still_counter, closed_eye_counter, sway_counter, excluded_this_frame, stutter_counter

    def _check_folded(
        self, landmarks, keypoints, prev_landmarks, prev_keypoints, frame_h
    ):
        """
        Rule 1 - 骨架扭曲检测：
        只抓真正的骨架数据异常，不误判正常动作。
        
        正常动作（不算折叠）：
        ✅ 正常弯手（摸脸、挠头、挥手）
        ✅ 转身/侧身（单侧关键点部分不可见）
        ✅ 抬腿/弯腿（走路、蹲下）
        ✅ 手臂自然下垂（三点共线）
        
        真正的骨架异常（才剪掉）：
        ❌ 单个关键点在一帧内跳跃超过身体高度的40%（骨架数据飞了）
        ❌ 双手同时举到头两侧极近位置且角度极小（<20°）— 蜷缩
        ❌ 四肢检测结果极不稳定（连续帧之间忽隐忽现）
        """
        threshold = self.cfg["fold_angle_threshold"]
        folded_limbs = 0

        # 安全检查：确保有足够的关键点数据
        if not keypoints or not prev_keypoints:
            return False

        # ── 帧间跳跃检测（主要检测骨架数据异常）──
        # 如果关键点在一帧内位移过大，说明骨架检测出了幻觉
        if prev_keypoints and len(prev_keypoints) > LANDMARK_NOSE:
            # 检查躯干关键点的帧间位移（鼻子、双肩、双髋）
            body_points = [LANDMARK_NOSE, LANDMARK_LEFT_SHOULDER, 
                          LANDMARK_RIGHT_SHOULDER, LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP]
            jumps = 0
            for idx in body_points:
                if idx < len(keypoints) and idx < len(prev_keypoints):
                    d = _pixel_distance(keypoints[idx], prev_keypoints[idx])
                    if d is not None and d > frame_h * 0.40:
                        jumps += 1
            # 躯干5个点中3个以上都在跳跃 → 骨架数据坏了
            if jumps >= 3:
                return True
        
        # ── 极端的蜷缩姿态检测 ──
        # 只有在双手都举到肩膀以上，且手肘弯到极限角度，才可能是真正扭曲
        for side in [
            (LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_ELBOW, LANDMARK_LEFT_WRIST),
            (LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_ELBOW, LANDMARK_RIGHT_WRIST),
        ]:
            shoulder, elbow, wrist = side
            if max(shoulder, elbow, wrist) < len(keypoints):
                shoulder_y = keypoints[shoulder][1]
                wrist_y = keypoints[wrist][1]
                # 只有手腕举到肩膀以上才检查
                if wrist_y < shoulder_y:
                    angle = _angle_between(
                        keypoints[shoulder],
                        keypoints[elbow],
                        keypoints[wrist],
                    )
                    # 角度极小（<20°）且双手都这样才是真蜷缩
                    if angle is not None and angle < 20:
                        folded_limbs += 1

        # 双手同时极度蜷缩才判定
        if folded_limbs >= 2:
            return True

        return False

    def _check_incomplete(self, landmarks, body_type):
        """
        Rule 2 - 不完整检测（放宽版）：
        不再强求双肩同时可见（允许侧身时只看到一侧），
        不再要求脚踝（很多画面边缘刚好切掉脚踝）。
        
        放宽后要求：
        - 鼻(0) + 至少一肩(11或12)
        - full：再加至少一髋(23或24)
        - half：再加至少一腕(15或16)

        好帧不改动，只放宽裁切标准。
        """
        conf = self.cfg["confidence_threshold"]

        def _visible(idx):
            return (
                idx < len(landmarks)
                and landmarks[idx].visibility >= conf
            )

        # 必选项：鼻
        if not _visible(LANDMARK_NOSE):
            return True

        # 至少一肩（允许侧身时一侧被挡）
        if not _visible(LANDMARK_LEFT_SHOULDER) and not _visible(LANDMARK_RIGHT_SHOULDER):
            return True

        if body_type == "full":
            # 至少一髋（允许侧身时一侧被挡）
            if not _visible(LANDMARK_LEFT_HIP) and not _visible(LANDMARK_RIGHT_HIP):
                return True
            # ⚠️ 不再要求脚踝（画面边缘容易切到）
        else:  # half
            # 至少一腕
            if not _visible(LANDMARK_LEFT_WRIST) and not _visible(LANDMARK_RIGHT_WRIST):
                return True

        return False

    def _check_still(
        self,
        landmarks,
        keypoints,
        prev_landmarks,
        prev_keypoints,
        still_counter,
        stillness_max_frames,
    ):
        """
        Rule 3 - 静止检测：
        计算手腕(15,16)和鼻子(0)的帧间位移
        """
        threshold = self.cfg["stillness_threshold"]

        if prev_keypoints and len(prev_keypoints) >= len(keypoints):
            # 计算鼻子、左右手腕的帧间位移
            displacements = []
            for idx in [LANDMARK_NOSE, LANDMARK_LEFT_WRIST, LANDMARK_RIGHT_WRIST]:
                if idx < len(keypoints) and idx < len(prev_keypoints):
                    d = _pixel_distance(keypoints[idx], prev_keypoints[idx])
                    if d is not None:
                        displacements.append(d)

            if displacements:
                avg_disp = sum(displacements) / len(displacements)
                if avg_disp < threshold:
                    still_counter += 1
                else:
                    still_counter = 0
            else:
                still_counter += 1
        else:
            still_counter += 1

        if still_counter >= stillness_max_frames:
            return True, still_counter

        return False, still_counter

    def _sliding_window_smooth(self, ratings, window_radius=2):
        """
        使用滑动窗口平滑评估结果：
        如果窗口中的 bad 帧占多数，当前帧标记 bad

        Args:
            ratings: list of "good" | "folded" | "incomplete" | "still"
            window_radius: 窗口半径（帧数），总窗口大小 = 2*radius + 1

        Returns:
            list[bool]: True=好帧
        """
        n = len(ratings)
        result = [True] * n

        for i in range(n):
            start = max(0, i - window_radius)
            end = min(n, i + window_radius + 1)

            bad_count = 0
            total = 0
            for j in range(start, end):
                total += 1
                if ratings[j] != "good":
                    bad_count += 1

            # 如果多数是 bad → bad
            if bad_count > total / 2:
                result[i] = False

        return result

    # ══════════════════════════════════════════════════════════════
    # Phase 3.5: 保留规则（姿态1评分 + 手部动作增强）
    # ══════════════════════════════════════════════════════════════

    def _calc_gesture_bonus(self, keypoints, frame_h, prev_keypoints=None):
        """
        手部动作加分（0.0~0.3）

        加分规则：
        1. 手腕y < 肩膀y（手腕在肩膀以上）-> +0.1  # 手抬起来了
        2. abs(左腕x - 右腕x) > 肩宽 -> +0.1        # 手张开了
        3. 手腕帧间位移 > 20px -> +0.1               # 手在动

        注意：如果 keypoints 长度不够，跳过对应加分项

        Args:
            keypoints: list of (x, y) — 当前帧所有关键点像素坐标
            frame_h: 帧高度（像素）
            prev_keypoints: list of (x, y) — 上一帧所有关键点像素坐标，可选

        Returns:
            float: bonus (0.0~0.3)
        """
        bonus = 0.0

        if not keypoints or len(keypoints) <= max(LANDMARK_RIGHT_WRIST, LANDMARK_RIGHT_SHOULDER, LANDMARK_LEFT_SHOULDER):
            return bonus

        left_wrist = keypoints[LANDMARK_LEFT_WRIST] if len(keypoints) > LANDMARK_LEFT_WRIST else None
        right_wrist = keypoints[LANDMARK_RIGHT_WRIST] if len(keypoints) > LANDMARK_RIGHT_WRIST else None
        left_shoulder = keypoints[LANDMARK_LEFT_SHOULDER] if len(keypoints) > LANDMARK_LEFT_SHOULDER else None
        right_shoulder = keypoints[LANDMARK_RIGHT_SHOULDER] if len(keypoints) > LANDMARK_RIGHT_SHOULDER else None

        # ── 条件1: 手腕抬过肩膀（左右任一手腕抬过肩膀即 +0.1）──
        wrist_above_shoulder = False
        if left_wrist and left_shoulder:
            if left_wrist[1] < left_shoulder[1]:  # y越小越靠上
                wrist_above_shoulder = True
        if right_wrist and right_shoulder:
            if right_wrist[1] < right_shoulder[1]:
                wrist_above_shoulder = True
        if wrist_above_shoulder:
            bonus += 0.1

        # ── 条件2: abs(左腕x - 右腕x) > 肩宽 ──
        if left_wrist and right_wrist and left_shoulder and right_shoulder:
            wrist_x_span = abs(left_wrist[0] - right_wrist[0])
            shoulder_width_x = abs(left_shoulder[0] - right_shoulder[0])
            if wrist_x_span > shoulder_width_x:
                bonus += 0.1

        # ── 条件3: 手腕帧间位移 > 20px ──
        movement_threshold = self.cfg.get("hand_wrist_movement_threshold", 20)
        if prev_keypoints and len(prev_keypoints) > LANDMARK_RIGHT_WRIST:
            for wrist_idx in [LANDMARK_LEFT_WRIST, LANDMARK_RIGHT_WRIST]:
                if wrist_idx < len(keypoints) and wrist_idx < len(prev_keypoints):
                    d = _pixel_distance(keypoints[wrist_idx], prev_keypoints[wrist_idx])
                    if d is not None and d > movement_threshold:
                        bonus += 0.1
                        break  # 只要有一个手腕满足条件即可

        return min(bonus, 0.3)

    def _calc_pose_score(self, keypoints, prev_keypoints=None, frame_h=None):
        """
        计算当前帧与"姿态1"（直立口播/半身演示）的匹配度。

        输入: keypoints (33个归一化关键点的像素坐标列表，每个元素为(x, y))
        输出: score (0.0~1.0) — 1.0=完美匹配姿态1

        v2 更新：加入手部动作加分
        score = 0.5 * coord_match_rate + 0.3 * angle_match_rate + 0.2 * gesture_bonus
        """
        if not keypoints or len(keypoints) < max(POSE_RANGES.keys()) + 1:
            return 0.0

        # 提取关键点
        def _kp(idx):
            if idx < len(keypoints) and keypoints[idx] is not None:
                return keypoints[idx]
            return None

        # 计算归一化参数
        hip_l = _kp(LANDMARK_LEFT_HIP)    # 23
        hip_r = _kp(LANDMARK_RIGHT_HIP)   # 24
        shoulder_l = _kp(LANDMARK_LEFT_SHOULDER)    # 11
        shoulder_r = _kp(LANDMARK_RIGHT_SHOULDER)   # 12

        if not all(p is not None for p in [hip_l, hip_r, shoulder_l, shoulder_r]):
            return 0.0

        # 原点 = 双髋中心
        origin_x = (hip_l[0] + hip_r[0]) / 2.0
        origin_y = (hip_l[1] + hip_r[1]) / 2.0

        # 缩放单位 = 肩宽
        shoulder_width = math.hypot(
            shoulder_l[0] - shoulder_r[0],
            shoulder_l[1] - shoulder_r[1],
        )
        if shoulder_width < 1e-6:
            return 0.0

        # ── 坐标匹配 ──
        coord_hits = 0
        coord_total = 0

        for lm_idx, ranges in POSE_RANGES.items():
            p = _kp(lm_idx)
            if p is None:
                continue

            # 归一化
            nx = (p[0] - origin_x) / shoulder_width
            ny = (p[1] - origin_y) / shoulder_width

            x_range, y_range = ranges

            # 如果该关键点没有定义范围（如脚踝），跳过
            if x_range[0] is None or x_range[1] is None:
                continue
            if y_range[0] is None or y_range[1] is None:
                continue

            coord_total += 1
            # 检查是否在范围内
            if (x_range[0] <= nx <= x_range[1]) and (y_range[0] <= ny <= y_range[1]):
                coord_hits += 1

        coord_match_rate = coord_hits / max(coord_total, 1)

        # ── 角度匹配 ──
        angle_match_rate = 0.0
        angle_checks = 0

        # 左肘角：左肩-左肘-左腕
        l_shoulder = _kp(LANDMARK_LEFT_SHOULDER)
        l_elbow = _kp(LANDMARK_LEFT_ELBOW)
        l_wrist = _kp(LANDMARK_LEFT_WRIST)
        if all(p is not None for p in [l_shoulder, l_elbow, l_wrist]):
            l_elbow_angle = _angle_between(l_shoulder, l_elbow, l_wrist)
            if l_elbow_angle is not None:
                angle_checks += 1
                diff = abs(l_elbow_angle - POSE_ELBOW_ANGLE)
                if diff <= POSE_ANGLE_TOLERANCE:
                    angle_match_rate += 1.0 - (diff / POSE_ANGLE_TOLERANCE) * 0.5

        # 右肘角：右肩-右肘-右腕
        r_shoulder = _kp(LANDMARK_RIGHT_SHOULDER)
        r_elbow = _kp(LANDMARK_RIGHT_ELBOW)
        r_wrist = _kp(LANDMARK_RIGHT_WRIST)
        if all(p is not None for p in [r_shoulder, r_elbow, r_wrist]):
            r_elbow_angle = _angle_between(r_shoulder, r_elbow, r_wrist)
            if r_elbow_angle is not None:
                angle_checks += 1
                diff = abs(r_elbow_angle - POSE_ELBOW_ANGLE)
                if diff <= POSE_ANGLE_TOLERANCE:
                    angle_match_rate += 1.0 - (diff / POSE_ANGLE_TOLERANCE) * 0.5

        # 左膝角：左髋-左膝-左踝
        l_hip = _kp(LANDMARK_LEFT_HIP)
        l_knee = _kp(LANDMARK_LEFT_KNEE)
        l_ankle = _kp(LANDMARK_LEFT_ANKLE)
        if all(p is not None for p in [l_hip, l_knee, l_ankle]):
            l_knee_angle = _angle_between(l_hip, l_knee, l_ankle)
            if l_knee_angle is not None:
                angle_checks += 1
                diff = abs(l_knee_angle - POSE_KNEE_ANGLE)
                if diff <= POSE_ANGLE_TOLERANCE:
                    angle_match_rate += 1.0 - (diff / POSE_ANGLE_TOLERANCE) * 0.5

        # 右膝角：右髋-右膝-右踝
        r_hip = _kp(LANDMARK_RIGHT_HIP)
        r_knee = _kp(LANDMARK_RIGHT_KNEE)
        r_ankle = _kp(LANDMARK_RIGHT_ANKLE)
        if all(p is not None for p in [r_hip, r_knee, r_ankle]):
            r_knee_angle = _angle_between(r_hip, r_knee, r_ankle)
            if r_knee_angle is not None:
                angle_checks += 1
                diff = abs(r_knee_angle - POSE_KNEE_ANGLE)
                if diff <= POSE_ANGLE_TOLERANCE:
                    angle_match_rate += 1.0 - (diff / POSE_ANGLE_TOLERANCE) * 0.5

        if angle_checks > 0:
            angle_match_rate /= angle_checks

        # ── 手部动作加分（v2 新增） ──
        gesture_bonus = self._calc_gesture_bonus(keypoints, frame_h or 720, prev_keypoints)

        # ── 综合评分（v2 更新权重） ──
        coord_w = self.cfg.get("pose_coord_weight", 0.5)
        angle_w = self.cfg.get("pose_angle_weight", 0.3)
        gesture_w = self.cfg.get("pose_gesture_weight", 0.2)

        # 如果没角度数据，降低角度权重
        if angle_checks == 0:
            score = coord_match_rate * (coord_w + angle_w) + gesture_bonus * gesture_w
        else:
            score = coord_match_rate * coord_w + angle_match_rate * angle_w + gesture_bonus * gesture_w

        return max(0.0, min(1.0, score))

    def _compute_pose_scores(self, frame_data, fps):
        """
        计算所有帧的姿态1评分

        对每帧调用 _calc_pose_score，跳过帧用最近结果插值。

        Args:
            frame_data: list[dict] — _scan_video 的输出
            fps: 帧率

        Returns:
            list[float]: 每帧的 pose score (0.0~1.0)
        """
        n = len(frame_data)
        scores = [0.0] * n
        last_score = 0.0
        prev_kp = None
        frame_h = frame_data[0].get("frame_h", 720) if frame_data else 720

        for i, f in enumerate(frame_data):
            kp = f.get("keypoint_positions", [])
            if kp and len(kp) >= LANDMARK_RIGHT_ANKLE + 1:
                # 有完整骨架检测结果 → 实际计算（含手部动作加分）
                score = self._calc_pose_score(kp, prev_kp, frame_h)
                scores[i] = score
                last_score = score
                prev_kp = kp
            else:
                # 跳过的帧或检测失败的帧 → 用最近值插值
                scores[i] = last_score

        return scores

    def _calc_segment_pose_score(self, start_frame, end_frame, pose_scores):
        """
        计算一段连续帧的"姿态1平均命中率"

        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引（不含）
            pose_scores: list[float] — 每帧的姿态评分

        Returns:
            float: 该段落的平均 pose score (0.0~1.0)
        """
        if start_frame >= end_frame:
            return 0.0

        segment_scores = pose_scores[start_frame:end_frame]
        if not segment_scores:
            return 0.0

        return sum(segment_scores) / len(segment_scores)

    def _get_frame_bbox_area(self, fd_entry):
        """计算一帧的骨架bbox面积（肩宽×躯干高）

        Args:
            fd_entry: 帧数据字典（frame_data中的一项）

        Returns:
            float: bbox面积，若无完整骨架数据则返回0
        """
        kp = fd_entry.get('keypoint_positions', {})
        # 支持 list 和 dict 两种格式
        if isinstance(kp, list):
            if len(kp) > 24 and len(kp) > 11:
                if kp[11] is not None and kp[12] is not None and kp[23] is not None and kp[24] is not None:
                    shoulder_w = abs(kp[12][0] - kp[11][0])
                    torso_h = abs((kp[11][1] + kp[12][1])/2 - (kp[23][1] + kp[24][1])/2)
                    return shoulder_w * torso_h
        elif isinstance(kp, dict):
            if 11 in kp and 12 in kp and 23 in kp and 24 in kp:
                shoulder_w = abs(kp[12][0] - kp[11][0])
                torso_h = abs((kp[11][1] + kp[12][1])/2 - (kp[23][1] + kp[24][1])/2)
                return shoulder_w * torso_h
        return 0

    # ── Phase 4: 找"保留规则命中率最高"的连贯段 ─────────────────

    def _find_best_segment(self, good_frames, frame_data, fps, target_duration, pose_scores, scene_cuts=None, target_bbox_size=None):
        """
        找"保留规则命中率最高"的连贯段。

        三段式：
        1. 先按场景切分（scene_cuts），在每个场景内部找好帧连续段
        2. 对每段计算 _calc_segment_pose_score()
        3. 筛选长度 >= target_duration 的段
        4. 选 pose_score 最高的一段
        5. 如果没有，选最长的段

        v2 关键：选段不跨场景

        Args:
            good_frames: list[bool] — 剔除规则标记的好/坏帧
            frame_data: list[dict] — 帧数据
            fps: 帧率
            target_duration: 目标时长（秒）
            pose_scores: list[float] — 每帧的姿态1评分
            scene_cuts: list[tuple] — 场景起止帧 [(start, end), ...]

        Returns:
            (start_frame, end_frame) or None
        """
        target_frames = int(target_duration * fps)
        n = len(good_frames)

        # Step 1: 确定场景边界
        if scene_cuts and len(scene_cuts) > 0:
            scenes = scene_cuts
        else:
            scenes = [(0, n - 1)]

        # Step 2: 在每个场景内部找好帧连续段
        segments = []  # [(start, end), ...]
        for scene_start, scene_end in scenes:
            # 将 scene_end 转换为 exclusive（方便计算）
            scene_end_ex = scene_end + 1
            # 在这个场景内找连续 good 段
            in_segment = False
            seg_start = 0
            for i in range(scene_start, scene_end_ex):
                if i >= n:
                    break
                is_good = good_frames[i]
                if is_good and not in_segment:
                    seg_start = i
                    in_segment = True
                elif not is_good and in_segment:
                    segments.append((seg_start, i))
                    in_segment = False
            if in_segment:
                segments.append((seg_start, min(scene_end_ex, n)))

        if not segments:
            return None

        # Step 3: 计算每段的姿态评分
        scored_segments = []
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < 1:
                continue
            avg_pose = self._calc_segment_pose_score(seg_start, seg_end, pose_scores)

            # 计算该段的平均bbox面积相似度
            bbox_score = 1.0  # 无参考值时不影响
            if target_bbox_size and target_bbox_size > 0:
                bbox_areas = []
                for f_idx in range(seg_start, min(seg_end, len(frame_data))):
                    area = self._get_frame_bbox_area(frame_data[f_idx])
                    if area > 0:
                        bbox_areas.append(area)
                avg_bbox = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0
                if avg_bbox > 0:
                    ratio = avg_bbox / target_bbox_size
                    if ratio > 1.0:
                        ratio = 1.0 / ratio
                    bbox_score = ratio  # 0~1, 越大越相似
                else:
                    bbox_score = 0.5  # 无骨架数据时给中间分

            # 综合评分：pose_score (70%) + bbox_score (30%)
            combined_score = avg_pose * 0.7 + bbox_score * 0.3

            scored_segments.append({
                "start": seg_start,
                "end": seg_end,
                "len": seg_len,
                "duration": seg_len / fps,
                "pose_score": avg_pose,
                "bbox_score": bbox_score,
                "combined_score": combined_score,
            })

        # 打印所有段信息
        print(f"\n[Phase 4-保留规则] 场景数={len(scenes)}, 好段数={len(scored_segments)}:")
        for s in scored_segments:
            flag = "✓" if s["len"] >= target_frames else "✗"
            print(f"  {flag} 帧 {s['start']}-{s['end']} "
                  f"({s['duration']:.1f}s, pose={s['pose_score']:.3f}, "
                  f"bbox={s['bbox_score']:.3f}, combined={s['combined_score']:.3f})")

        if not scored_segments:
            return None

        # Step 4: 筛选长度足够的段
        qualified = [s for s in scored_segments if s["len"] >= target_frames]

        if qualified:
            # 从足够长的段中选综合评分最高的一段
            best = max(qualified, key=lambda s: s.get('combined_score', s['pose_score']))
            print(f"\n[Phase 4] 选择: 帧 {best['start']}-{best['end']} "
                  f"(足够长, combined={best.get('combined_score', best['pose_score']):.3f})")

            # 从中间取 target_duration
            mid = (best["start"] + best["end"]) // 2
            half_target = target_frames // 2
            result_start = max(best["start"], mid - half_target)
            result_end = min(best["end"], result_start + target_frames)
            # 修正偏移
            if result_end - result_start < target_frames:
                result_start = max(best["start"], result_end - target_frames)

            return (result_start, result_end)

        else:
            # Step 5: 没有够长的段，放宽限制再找
            print(f"\n[Phase 4] ⚠️ 无足够长段({target_duration}s)，放宽骨架限制重找...")
            
            # 放宽策略：把所有有骨架的坏帧当作好帧（用户要求太短就放宽限制）
            relaxed_good = list(good_frames)
            for i, f in enumerate(frame_data):
                if not relaxed_good[i] and f.get("landmarks") is not None:
                    relaxed_good[i] = True
            
            # 用放宽后的标准重找
            relaxed_segments = []
            for scene_start, scene_end in scenes:
                scene_end_ex = scene_end + 1
                in_segment = False
                seg_start = 0
                for i in range(scene_start, scene_end_ex):
                    if i >= n:
                        break
                    is_good = relaxed_good[i]
                    if is_good and not in_segment:
                        seg_start = i
                        in_segment = True
                    elif not is_good and in_segment:
                        relaxed_segments.append((seg_start, i))
                        in_segment = False
                if in_segment:
                    relaxed_segments.append((seg_start, min(scene_end_ex, n)))
            
            if not relaxed_segments:
                # 彻底放宽：选最长连续有骨架的帧段
                print("[Phase 4] 仍无可用的好段，取最长有骨架段...")
                best_raw = None
                best_raw_len = 0
                for scene_start, scene_end in scenes:
                    start = None
                    for i in range(scene_start, min(scene_end + 1, n)):
                        if frame_data[i]["landmarks"] is not None:
                            if start is None:
                                start = i
                        else:
                            if start is not None:
                                seg_len = i - start
                                if seg_len > best_raw_len:
                                    best_raw_len = seg_len
                                    best_raw = (start, i)
                            start = None
                    if start is not None:
                        seg_len = min(scene_end + 1, n) - start
                        if seg_len > best_raw_len:
                            best_raw_len = seg_len
                            best_raw = (start, min(scene_end + 1, n))
                
                if best_raw:
                    longest = {"start": best_raw[0], "end": best_raw[1], 
                              "len": best_raw_len, "duration": best_raw_len / fps}
                else:
                    longest = max(scored_segments, key=lambda s: s["len"])
            else:
                # 从放宽后的段中选最长
                relaxed_scored = []
                for seg_start, seg_end in relaxed_segments:
                    seg_len = seg_end - seg_start
                    if seg_len < 1:
                        continue
                    avg_pose = self._calc_segment_pose_score(seg_start, seg_end, pose_scores)

                    # 计算bbox相似度（放宽路径也遵循）
                    bbox_score = 1.0
                    if target_bbox_size and target_bbox_size > 0:
                        bbox_areas = []
                        for f_idx in range(seg_start, min(seg_end, len(frame_data))):
                            area = self._get_frame_bbox_area(frame_data[f_idx])
                            if area > 0:
                                bbox_areas.append(area)
                        avg_bbox = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0
                        if avg_bbox > 0:
                            ratio = avg_bbox / target_bbox_size
                            if ratio > 1.0:
                                ratio = 1.0 / ratio
                            bbox_score = ratio
                        else:
                            bbox_score = 0.5

                    combined_score = avg_pose * 0.7 + bbox_score * 0.3

                    relaxed_scored.append({
                        "start": seg_start, "end": seg_end, "len": seg_len,
                        "duration": seg_len / fps, "pose_score": avg_pose,
                        "bbox_score": bbox_score, "combined_score": combined_score,
                    })
                
                # 优先选够target的，没有就选最长
                qualified = [s for s in relaxed_scored if s["len"] >= target_frames]
                if qualified:
                    longest = max(qualified, key=lambda s: s.get('combined_score', s['pose_score']))
                else:
                    longest = max(relaxed_scored, key=lambda s: s["len"])
            
            print(f"[Phase 4] 放宽后选择: 帧 {longest['start']}-{longest['end']} "
                  f"({longest['duration']:.1f}s)")
            
            # 从中间取 target_duration
            best = longest
            mid = (best["start"] + best["end"]) // 2
            half_target = target_frames // 2
            result_start = max(best["start"], mid - half_target)
            result_end = min(best["end"], result_start + target_frames)
            if result_end - result_start < target_frames:
                result_start = max(best["start"], result_end - target_frames)
            
            return (result_start, result_end)

    # ── Phase 5: 输出视频 ────────────────────────────────────

    def _output_segment(
        self, input_path, output_path, start_frame, end_frame, fps
    ):
        """
        用 OpenCV 从源视频中提取帧序列并写出

        使用 'mp4v' 编码，兼容性最好
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {input_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

        if not out.isOpened():
            cap.release()
            raise IOError(f"无法创建输出视频: {output_path}")

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if start_frame <= frame_idx < end_frame:
                    out.write(frame)

                frame_idx += 1
                if frame_idx >= end_frame:
                    break
        finally:
            cap.release()
            out.release()

        # 验证输出
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise IOError(f"输出视频文件无效: {output_path}")

        out_cap = cv2.VideoCapture(output_path)
        out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_cap.release()
        print(f"[Phase 5] 输出: {output_path} ({out_frames}帧, {out_frames / fps:.1f}s)")

        # 尝试复制音轨（OpenCV不处理音频）
        try:
            import subprocess
            temp_path = output_path + ".tmp.mp4"
            os.rename(output_path, temp_path)
            subprocess.run([
                "ffmpeg", "-i", temp_path, "-i", input_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ], capture_output=True, timeout=30)
            os.remove(temp_path)
        except Exception:
            # 没有音轨或ffmpeg不可用，已存在的 output_path 就是用 OpenCV 生成的无音轨版本
            pass

    def _concat_videos(self, video_paths, output_path):
        """
        用 OpenCV 流式拼接多个视频文件（边读边写，不缓存所有帧到内存）
        """
        if not video_paths:
            raise ValueError("没有视频可拼接")

        # 先打开第一个视频，确定输出参数
        fps = 30
        size = None

        caps = []
        try:
            for vpath in video_paths:
                cap = cv2.VideoCapture(vpath)
                if not cap.isOpened():
                    print(f"  [⚠️] 跳过无法打开的视频: {vpath}")
                    continue

                if size is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    size = (frame_w, frame_h)

                caps.append(cap)

            if not caps:
                raise ValueError("没有有效视频可拼接")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, size)
            if not out.isOpened():
                raise IOError(f"无法创建输出视频: {output_path}")

            total_frames = 0
            try:
                for cap in caps:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # 如果大小不一致，调整大小
                        if (frame.shape[1], frame.shape[0]) != size:
                            frame = cv2.resize(frame, size)
                        out.write(frame)
                        total_frames += 1
            finally:
                out.release()

            print(f"  [拼接] 输出: {output_path} ({total_frames}帧, "
                  f"{total_frames / fps:.1f}s)")
        finally:
            for cap in caps:
                cap.release()
