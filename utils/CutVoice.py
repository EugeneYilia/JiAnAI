from pydub import AudioSegment
import numpy as np
from scipy.ndimage import uniform_filter1d

def trim_tail_by_energy_and_gradient(
    audio_path,
    threshold_db=-35,
    min_tail_ms=500,
    frame_ms=30,
    gradient_thresh=0.5,
    fade_out_ms=150
):
    audio = AudioSegment.from_file(audio_path)
    samples = np.array(audio.get_array_of_samples())
    frame_len = int(audio.frame_rate * frame_ms / 1000)

    energies = []
    for i in range(0, len(samples), frame_len):
        frame = samples[i:i+frame_len]
        if len(frame) == 0:
            continue
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        db = 20 * np.log10(rms + 1e-10)
        energies.append(db)

    energies = np.array(energies)
    smoothed = uniform_filter1d(energies, size=3)

    # 计算梯度（能量变化率）
    gradient = np.abs(np.diff(smoothed))

    # 反向寻找“最近一次稳定语音段”
    end_idx = len(smoothed) - 1
    for i in reversed(range(len(gradient))):
        if smoothed[i] > threshold_db and gradient[i] > gradient_thresh:
            end_idx = i
            break

    end_ms = int(end_idx * frame_ms + min_tail_ms)
    end_ms = min(len(audio), end_ms)

    trimmed = audio[:end_ms].fade_out(fade_out_ms)
    trimmed.export(audio_path, format="wav")
    print(f"[优化剪辑] 保留到 {end_ms/1000:.2f}s，原始长度 {len(audio)/1000:.2f}s")
