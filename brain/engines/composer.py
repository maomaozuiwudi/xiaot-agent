"""视频合成引擎 — 从旧项目 video_composer.py 搬运核心函数"""
import os
from moviepy import (
    ImageClip, VideoFileClip, AudioFileClip,
    concatenate_videoclips, CompositeAudioClip,
    vfx, afx,
)


def compose_mixed(shots, output_path, voiceover_path=None, bgm_path=None,
                  resolution=(1080, 1920), fps=30):
    """
    混合合成：图片+视频混排拼接

    Args:
        shots: [{"type": "image"|"video", "path": str, "duration": float}, ...]
        output_path: 输出MP4路径
        voiceover_path: 配音音频（可选）
        bgm_path: 背景音乐（可选）
        resolution: (宽, 高)，默认 1080x1920 竖屏
        fps: 帧率
    """
    # 前置校验
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            raise ValueError(f"shots[{i}] 不是字典")
        if s["type"] not in ("image", "video"):
            raise ValueError(f"shots[{i}] type 必须为 image 或 video")
        if not isinstance(s["duration"], (int, float)) or s["duration"] <= 0:
            raise ValueError(f"shots[{i}] duration 必须为正数")
        if not os.path.isfile(s["path"]):
            raise ValueError(f"shots[{i}] 文件不存在: {s['path']}")

    clips = []
    for s in shots:
        if s["type"] == "image":
            clip = ImageClip(s["path"]).with_duration(s["duration"])
            clip = clip.with_effects([vfx.Resize((resolution[0], resolution[1]))])
        else:
            clip = VideoFileClip(s["path"], audio=False)
            clip = clip.with_effects([vfx.Resize((resolution[0], resolution[1]))])
            actual_dur = min(clip.duration, s["duration"])
            if clip.duration > actual_dur:
                clip = clip.subclipped(0, actual_dur)
            elif clip.duration < actual_dur:
                clip = clip.with_effects([vfx.Loop(duration=actual_dur)])
        clips.append(clip)

    final = concatenate_videoclips(clips, method="chain")
    video_duration = final.duration

    # 加音频
    audio_tracks = []
    if voiceover_path and os.path.isfile(voiceover_path):
        voice = AudioFileClip(voiceover_path)
        if voice.duration > video_duration:
            voice = voice.subclipped(0, video_duration)
        audio_tracks.append(voice)
    if bgm_path and os.path.isfile(bgm_path):
        bgm = AudioFileClip(bgm_path)
        if bgm.duration < 1.0:
            bgm.close()
        else:
            if bgm.duration < video_duration:
                bgm = bgm.loop(duration=video_duration)
            else:
                bgm = bgm.subclipped(0, video_duration)
            bgm = bgm.with_effects([afx.MultiplyVolume(0.15)])
            audio_tracks.append(bgm)

    if audio_tracks:
        final_audio = CompositeAudioClip(audio_tracks)
        final = final.with_audio(final_audio)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")

    for c in clips:
        try:
            c.close()
        except Exception:
            pass
    try:
        final.close()
    except Exception:
        pass

    return output_path
