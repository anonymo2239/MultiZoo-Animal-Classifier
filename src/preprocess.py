import cv2
import os
import numpy as np
import shutil
import random

class DataPreprocessor:
    def __init__(self, folder_dir):
        self.folder_dir = folder_dir

    def resize_images(self, width=224, height=224):
        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        img = cv2.imread(image_path)
                        if img is None:
                            continue
                        resized = cv2.resize(img, (width, height))
                        cv2.imwrite(image_path, resized)
        print("✔ Yeniden boyutlandırma işlemi tamamlandı.")

    def normalize_images(self):
        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                        normalized = normalized.astype(np.uint8)
                        cv2.imwrite(image_path, normalized)
        print("✔ Normalize işlemi tamamlandı.")

    def histogram_equalization(self):
        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                        cv2.imwrite(image_path, equalized)
        print("✔ Histogram eşitleme tamamlandı.")

    def detect_low_quality_images(self, blur_thresh=100.0, edge_thresh=0.35, move_to='low_quality'):
        os.makedirs(move_to, exist_ok=True)

        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        edges = cv2.Canny(gray, 100, 200)
                        edge_density = np.sum(edges) / (edges.size * 255)

                        if blur_score < blur_thresh or edge_density > edge_thresh:
                            print(f"Düşük kalite: {image_path} | Blur: {blur_score:.2f}, Edge: {edge_density:.4f}")
                            target_path = os.path.join(move_to, f"{class_folder}_{image_file}")
                            shutil.copy(image_path, target_path)
        print("✔ Kalite taraması tamamlandı.")

    def augment_images(self, num_augmented=2):
        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        image = cv2.imread(image_path)
                        if image is None:
                            continue

                        h, w = image.shape[:2]

                        for i in range(num_augmented):
                            aug = image.copy()

                            # Flip
                            if np.random.rand() > 0.5:
                                aug = cv2.flip(aug, 1)
                            if np.random.rand() > 0.7:
                                aug = cv2.flip(aug, 0)

                            # Rotation
                            if np.random.rand() > 0.5:
                                angle = np.random.choice([90, -90])
                                center = (w // 2, h // 2)
                                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                aug = cv2.warpAffine(aug, M, (w, h))

                            # Scaling
                            scale_factor = np.random.uniform(0.85, 1.15)
                            scaled = cv2.resize(aug, None, fx=scale_factor, fy=scale_factor)
                            scaled = cv2.resize(scaled, (w, h))
                            aug = scaled

                            # Border
                            if np.random.rand() > 0.8:
                                aug = cv2.copyMakeBorder(aug, 10, 10, 10, 10, borderType=cv2.BORDER_REFLECT)
                                aug = cv2.resize(aug, (w, h))

                            # Gaussian Blur
                            if np.random.rand() > 0.7:
                                ksize = random.choice([3, 5])
                                aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

                            # Brightness change
                            if np.random.rand() > 0.6:
                                value = np.random.randint(-30, 30)
                                hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV)
                                v_channel = hsv[:, :, 2].astype(int) + value
                                hsv[:, :, 2] = np.clip(v_channel, 0, 255).astype(np.uint8)
                                aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                            new_filename = f"{os.path.splitext(image_file)[0]}_aug{i}.jpg"
                            new_path = os.path.join(class_path, new_filename)
                            cv2.imwrite(new_path, aug)
        print("✔ Veri çoğaltma (flip, rotate, scale, blur, border, brightness) tamamlandı.")


if __name__ == "__main__":
    folder_dir = "C:/train"
    dp = DataPreprocessor(folder_dir)
    dp.resize_images()
    dp.histogram_equalization()
    dp.normalize_images()
    dp.detect_low_quality_images()
    dp.augment_images(num_augmented=2)
