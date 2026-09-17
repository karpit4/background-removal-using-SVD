import numpy as np
import matplotlib.pyplot as plt
import os 
from algorithms import trunc_svd, rand_svd
from parser import load_input


def main():
    # --------------------------------------------------------
    # One experiment:
    # video -> matrix -> SVD algorithm -> background/foreground
    # --------------------------------------------------------

    path = ("videos/MSA") #<-- name of your file or directory

    arr = load_input(path)
    print("Video transformed into an array")
    nframes, size_h, size_w = arr.shape

    first_frame = arr[0]

    # Convert video to matrix M: pixels x frames.
    M = arr.reshape(nframes, size_h * size_w).T

    # Any SVD algorithm from algorithms.py can be used here.
    # To switch algorithms, replace only this line.
    
    # U, S, Vt = trunc_svd(M, r=1)
    U, S, Vt = rand_svd(M, r=1, oversampling=10)


    # Reconstruct the rank-1 background approximation.
    M_background = U @ np.diag(S) @ Vt

    background_frame = M_background[:, 0].reshape(size_h, size_w)
    foreground_frame = first_frame - background_frame

    # Show original frame, background and moving objects.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(first_frame, cmap="gray")
    axs[0].set_title("Исходное изображение")
    axs[0].axis("off")

    axs[1].imshow(background_frame, cmap="gray")
    axs[1].set_title("Фон")
    axs[1].axis("off")

    axs[2].imshow(foreground_frame, cmap="gray")
    axs[2].set_title("Движущиеся объекты")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
