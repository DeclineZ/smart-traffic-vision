"""
Realistic Synthetic Rain & Wet Road Augmentation Engine.
Simulates adverse precipitation conditions (rain streaks, atmospheric fog/mist,
and wet road surface darkening with specular sheen) for CCTV camera frames.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import cv2
import numpy as np


class SyntheticRainAugmentor:
    """
    Physically inspired Synthetic Rain Augmentor for traffic surveillance images.
    Applies:
    1. Directional rain streaks with motion blur
    2. Atmospheric mist / contrast attenuation
    3. Road surface darkening and specular reflection highlights
    """

    def __init__(
        self,
        slant_range: tuple[int, int] = (-20, 20),
        density_range: tuple[int, int] = (800, 2200),
        length_range: tuple[int, int] = (15, 35),
        fog_alpha: float = 0.15,
        wet_sheen: bool = True,
        seed: int | None = None,
    ):
        self.slant_range = slant_range
        self.density_range = density_range
        self.length_range = length_range
        self.fog_alpha = fog_alpha
        self.wet_sheen = wet_sheen
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _generate_rain_layer(
        self,
        shape: tuple[int, int, int],
        slant: int,
        length: int,
        num_drops: int,
    ) -> np.ndarray:
        """Draws directional semi-transparent rain streaks with motion blur."""
        h, w = shape[:2]
        rain_layer = np.zeros((h, w), dtype=np.uint8)

        # Compute dx, dy according to slant angle (degrees from vertical)
        angle_rad = np.radians(slant)
        dx = int(length * np.sin(angle_rad))
        dy = int(length * np.cos(angle_rad))
        if dy == 0:
            dy = length

        # Random drop origins
        xs = np.random.randint(0, w, size=num_drops)
        ys = np.random.randint(0, h, size=num_drops)

        for x, y in zip(xs, ys):
            x2 = np.clip(x + dx, 0, w - 1)
            y2 = np.clip(y + dy, 0, h - 1)
            # Vary drop brightness for depth realism
            intensity = random.randint(180, 255)
            cv2.line(rain_layer, (x, y), (x2, y2), intensity, thickness=random.choice([1, 1, 1, 2]))

        # Apply motion blur kernel along the streak direction
        kernel_size = max(3, length // 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(rain_layer, (kernel_size, kernel_size), 0)
        return blurred

    def _apply_fog_mist(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """Simulates atmospheric scattering and reduced contrast during downpours."""
        # Grayish-white atmospheric mist
        fog = np.full_like(image, 215, dtype=np.uint8)
        fogged = cv2.addWeighted(image, 1.0 - alpha, fog, alpha, 0)
        return fogged

    def _apply_wet_road_sheen(self, image: np.ndarray, road_roi_top: float = 0.35) -> np.ndarray:
        """
        Simulates wet pavement:
        1. Darkens lower road surface (diffuse reflection decreases when wet)
        2. Enhances bright specular highlights (standing water reflections)
        """
        h, w = image.shape[:2]
        y_start = int(h * road_roi_top)
        road_region = image[y_start:h, :].astype(np.float32)

        # 1. Darken wet asphalt slightly
        darkened_road = road_region * 0.88

        # 2. Specular highlight booster on bright areas (reflecting streetlights / sky)
        gray_road = cv2.cvtColor(road_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        specular_mask = (gray_road > 165).astype(np.float32)
        # Boost bright spots
        darkened_road += specular_mask[:, :, None] * 28.0

        image[y_start:h, :] = np.clip(darkened_road, 0, 255).astype(np.uint8)
        return image

    def augment(
        self,
        image: np.ndarray,
        intensity: str = "medium",
        slant: int | None = None,
    ) -> np.ndarray:
        """
        Applies complete synthetic rain transformation to a single BGR image.
        intensity: 'light', 'medium', or 'heavy'
        """
        out = image.copy()
        h, w = out.shape[:2]

        # Determine intensity multipliers
        if intensity == "light":
            drop_mult = 0.6
            fog_mult = 0.6
            length_mult = 0.75
        elif intensity == "heavy":
            drop_mult = 1.6
            fog_mult = 1.4
            length_mult = 1.3
        else:  # medium
            drop_mult = 1.0
            fog_mult = 1.0
            length_mult = 1.0

        # 1. Apply wet road darkening and specular sheen
        if self.wet_sheen:
            out = self._apply_wet_road_sheen(out, road_roi_top=0.30)

        # 2. Apply atmospheric mist
        eff_fog = min(0.35, self.fog_alpha * fog_mult)
        if eff_fog > 0.01:
            out = self._apply_fog_mist(out, eff_fog)

        # 3. Generate rain streak layer
        if slant is None:
            slant = random.randint(self.slant_range[0], self.slant_range[1])
        base_drops = random.randint(self.density_range[0], self.density_range[1])
        num_drops = int(base_drops * (w * h / (640 * 360)) * drop_mult)
        length = int(random.randint(self.length_range[0], self.length_range[1]) * length_mult)

        rain_mask = self._generate_rain_layer((h, w, 3), slant=slant, length=length, num_drops=num_drops)

        # 4. Blend rain streaks onto image
        rain_color = cv2.cvtColor(rain_mask, cv2.COLOR_GRAY2BGR)
        # Additive blend with soft ceiling
        out = cv2.addWeighted(out, 1.0, rain_color, 0.45, 0)
        return out


def create_comparison_grid(
    original: np.ndarray,
    augmented: np.ndarray,
    title_left: str = "ORIGINAL (DRY)",
    title_right: str = "SYNTHETIC RAIN AUGMENTED",
) -> np.ndarray:
    """Concatenates original and augmented images with labeled headers."""
    h, w = original.shape[:2]
    # Header bar
    bar_h = 42
    canvas = np.zeros((h + bar_h, w * 2, 3), dtype=np.uint8)
    canvas[:bar_h, :] = (30, 30, 30)

    # Insert images
    canvas[bar_h:, :w] = original
    canvas[bar_h:, w:] = augmented

    # Text headers
    cv2.putText(canvas, title_left, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, title_right, (w + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 160, 50), 2, cv2.LINE_AA)

    # Divider line
    cv2.line(canvas, (w, 0), (w, h + bar_h), (80, 80, 80), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Synthetic Rain Augmentation Engine for Traffic Cameras.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image.")
    parser.add_argument("--video", type=str, default=None, help="Path to video to extract and augment a sample frame.")
    parser.add_argument("--intensity", type=str, default="medium", choices=["light", "medium", "heavy"], help="Rain intensity.")
    parser.add_argument("--slant", type=int, default=15, help="Slant angle in degrees (-30 to 30).")
    parser.add_argument("--output", type=str, default="output/synthetic_rain_sample.jpg", help="Output path.")
    parser.add_argument("--compare", action="store_true", help="Save side-by-side comparison.")
    parser.add_argument("--batch-dir", type=str, default=None, help="Batch augment directory of images.")
    parser.add_argument("--batch-out", type=str, default=None, help="Batch output directory.")
    parser.add_argument("--count", type=int, default=50, help="Number of images to augment in batch.")

    args = parser.parse_args()
    augmentor = SyntheticRainAugmentor()

    if args.image:
        img_p = Path(args.image)
        img = cv2.imread(str(img_p.resolve()))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_p}")
        aug = augmentor.augment(img, intensity=args.intensity, slant=args.slant)
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if args.compare:
            canvas = create_comparison_grid(img, aug)
            cv2.imwrite(str(out_p.resolve()), canvas)
        else:
            cv2.imwrite(str(out_p.resolve()), aug)
        print(f"[✔] Saved synthetic rain image to: {out_p.resolve()}")

    elif args.video:
        vid_p = Path(args.video)
        cap = cv2.VideoCapture(str(vid_p.resolve()))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {vid_p}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame from: {vid_p}")

        aug = augmentor.augment(frame, intensity=args.intensity, slant=args.slant)
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if args.compare:
            canvas = create_comparison_grid(frame, aug)
            cv2.imwrite(str(out_p.resolve()), canvas)
        else:
            cv2.imwrite(str(out_p.resolve()), aug)
        print(f"[✔] Saved synthetic rain comparison from video: {out_p.resolve()}")

    elif args.batch_dir and args.batch_out:
        src_dir = Path(args.batch_dir)
        dest_dir = Path(args.batch_out)
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
        sampled = random.sample(files, min(args.count, len(files)))

        print(f"[*] Batch augmenting {len(sampled)} images from {src_dir} -> {dest_dir}")
        for i, fp in enumerate(sampled, 1):
            img = cv2.imread(str(fp.resolve()))
            if img is None:
                continue
            aug = augmentor.augment(img, intensity=random.choice(["light", "medium", "heavy"]))
            out_file = dest_dir / f"syn_rain_{fp.stem}.jpg"
            cv2.imwrite(str(out_file), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[✔] Finished augmenting {len(sampled)} images into {dest_dir.resolve()}")


if __name__ == "__main__":
    main()
