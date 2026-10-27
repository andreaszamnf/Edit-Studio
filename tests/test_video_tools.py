import pytest

from conftest import wait
from edit_studio import media


@pytest.fixture
def clip(services, tmp_path):
    """A 2-second 320x240 test video with a tone, uploaded like a user file."""
    src = tmp_path / "clip.mp4"
    media.run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=2", "-f", "lavfi",
                      "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                      "-shortest", str(src)])
    return services.storage.add_upload(src, "clip.mp4")


def run(services, name, params):
    job = wait(services.jobs.submit(name, params), timeout=180)
    assert job.status == "done", (name, job.error_key, job.error)
    return [services.storage.get(i) for i in job.outputs]


def test_upload_is_probed(clip):
    assert clip.kind == "video" and (clip.width, clip.height) == (320, 240)
    assert abs(clip.duration - 2.0) < 0.2 and clip.has_audio and clip.thumb


def test_cut_speed_crop(services, clip):
    [out] = run(services, "video.transform", {"video": clip.id, "start": 0.5, "end": 1.5, "speed": 2.0, "aspect": "1:1"})
    assert out.width == out.height == 240
    assert abs(out.duration - 0.5) < 0.15 and out.has_audio


def test_reverse_rotate_mute_resize(services, clip):
    [out] = run(services, "video.transform", {"video": clip.id, "reverse": True, "rotate": 90, "mute": True, "height": 160})
    assert out.height == 160 and not out.has_audio


def test_gif_frame_and_join(services, clip):
    [gif] = run(services, "video.to_gif", {"video": clip.id, "fps": 8, "width": 160})
    assert gif.file.endswith(".gif") and gif.width == 160
    [frame] = run(services, "video.extract_frame", {"video": clip.id, "time": 1.0})
    assert frame.kind == "image" and (frame.width, frame.height) == (320, 240)
    [last] = run(services, "video.extract_frame", {"video": clip.id, "time": -1})
    assert last.kind == "image"
    [joined] = run(services, "video.concat", {"videos": [clip.id, clip.id]})
    assert abs(joined.duration - 4.0) < 0.3 and joined.has_audio


def test_smooth_motion_and_slowmo(services, clip):
    [smooth] = run(services, "video.interpolate", {"video": clip.id, "factor": 2})
    assert round(smooth.fps) == 48 and abs(smooth.duration - 2.0) < 0.3
    [slow] = run(services, "video.interpolate", {"video": clip.id, "factor": 2, "slowmo": True})
    assert abs(slow.duration - 4.0) < 0.4


def test_bad_inputs_fail_cleanly(services, clip):
    job = wait(services.jobs.submit("video.transform", {"video": "u-doesnotexist"}))
    assert job.status == "failed" and job.error_key == "input_missing"
