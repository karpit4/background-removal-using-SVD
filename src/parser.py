import moviepy.editor as mpe
import os
import glob
import numpy as np
from PIL import Image
import cv2


def frames_to_array(folder):
    """
    Loads a sequence of images from a folder
    and returns a grayscale array:
        (number of frames, height, width)

    Supported formats: .png, .jpg, .jpeg, .ppm
    """

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.ppm")

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))

    if not files:
        raise FileNotFoundError(f"В папке '{folder}' не найдено изображений.")

    files.sort()

    first = np.array(Image.open(files[0]).convert("L"))

    height, width = first.shape
    nframes = len(files)

    arr = np.zeros((nframes, height, width), dtype=float)

    for i, filename in enumerate(files):
        frame = np.array(Image.open(filename).convert("L"))

        if frame.shape != (height, width):
            raise ValueError(
                f"Size of frame  {filename} differs from frame1: "
                f"{frame.shape} != {(height, width)}"
            )

        arr[i] = frame

    print(f"image size: {width} x {height},")
    print(f"number of frames: {nframes}")

    return arr

def video_to_array(filename):
    video = mpe.VideoFileClip(filename)
    """Convert a video to a grayscale array: frames x height x width."""
    duration = int(video.duration)
    nframes = int(video.fps * video.duration)
    size_w, size_h = video.size

    arr = np.zeros((nframes, size_h, size_w))

    for i in range(nframes):
        arr[i] = video.get_frame(i / nframes * duration)[:, :, 0].astype(float)

    print(f"image size: {size_w} x {size_h},")
    print(f"number of frames: {nframes}")

    return arr


def load_input(path):
    """
    Loads data from a video file or a folder containing frames.

    If `path` points to a file -> `video_to_array()` is used.
    If `path` points to a folder -> `frames_to_array()` is used.

    Returns:
        array — a sequence of frames.
    """

    if os.path.isfile(path):
        return video_to_array(path)

    elif os.path.isdir(path):
        return frames_to_array(path)

    else:
        raise FileNotFoundError(
            f"Path '{path}' not found "
        )




def array_to_video(array, output_path, fps):
    """
    Numpy array --> videofile

    Parameters
    ----------
    array : np.ndarray
        (frames, height, width)

    output_path : str
        "results/"

    fps : float
        Частота кадров исходного видео.
    """

    array = np.clip(array, 0, 255).astype(np.uint8)

    nframes, height, width = array.shape

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=False
    )

    if not writer.isOpened():
        raise IOError(f"Failed to create a video: {output_path}")

    for frame in array:
        writer.write(frame)

    writer.release()