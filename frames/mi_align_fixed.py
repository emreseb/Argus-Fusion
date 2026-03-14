import cv2
import numpy as np
from scipy import optimize
import argparse


def preprocess(img, target_size=1024):
    """Convert to grayscale, apply CLAHE, and resize."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    h, w = img.shape
    scale = target_size / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img


def joint_histogram(img1, img2, bins=64):
    hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    return hgram + 1e-9   # avoid zeros


def mutual_information(hgram):
    pxy = hgram / np.sum(hgram)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]
    nz = pxy > 0

    return np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))


def mi_cost(params, img_m, img_r):
    tx, ty, theta = params
    h, w = img_r.shape

    M = cv2.getRotationMatrix2D((w//2, h//2), theta, 1.0)
    M[:, 2] += (tx, ty)

    warped = cv2.warpAffine(img_m, M, (w, h))

    hgram = joint_histogram(img_r, warped)
    return -mutual_information(hgram)   # minimize negative MI


def align_mi(img_m, img_r):
    initial = np.array([0, 0, 0], dtype=float)
    result = optimize.minimize(mi_cost, initial, args=(img_m, img_r),
                               method="Nelder-Mead",
                               options={"maxiter": 200})

    return result.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--mov", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ref = cv2.imread(args.ref)
    mov = cv2.imread(args.mov)

    ref_p = preprocess(ref)
    mov_p = preprocess(mov)

    tx, ty, theta = align_mi(mov_p, ref_p)

    print(f"Alignment params: tx={tx}, ty={ty}, theta={theta}")

    # Apply to original resolution moving image
    h, w, _ = ref.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), theta, 1.0)
    M[:, 2] += (tx, ty)

    aligned = cv2.warpAffine(mov, M, (w, h))
    cv2.imwrite(args.out, aligned)


if __name__ == "__main__":
    main()
