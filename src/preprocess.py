import cv2
import os
import numpy as np
import shutil

folder_dir = "C:/train"

class DataPreprocessor():
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
                        resized = cv2.resize(img, (width, height))
                        cv2.imwrite(image_path, resized)
        print("✔ Yeniden boyutlandırma işlemi tamamlandı.")

    def normalize_images(self): # burası optimize edilebilir.
        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        image = cv2.imread(image_path)
                        normalized_color_image = cv2.normalize(
                            image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                        cv2.imwrite(image_path, normalized_color_image)
        print("✔ Normalize işlemi tamamlandı.")
                        
    def histogram_equalization(self): # histogram derecesi optimize edilebilir.
        for class_folder in os.listdir(self.folder_dir):
            class_path = os.path.join(self.folder_dir, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_file)
                        image = cv2.imread(image_path)
                        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

                        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                        cv2.imwrite(image_path, equalized)
        print("✔ Histogram eşitleme tamamlandı.")


    def detect_low_quality_images(self, blur_thresh=100.0, edge_thresh=250.0, move_to='low_quality'):
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

                        # 1. Bulanıklık skoru (Laplacian variance)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                        # 2. Kenar yoğunluğu (Canny edge + ortalama piksel yoğunluğu)
                        edges = cv2.Canny(gray, 100, 200)
                        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

                        # Kalite eşiği altında mı?
                        if blur_score < blur_thresh or edge_density > edge_thresh:
                            print(f"Düşük kalite: {image_path} | Blur: {blur_score:.2f}, Edge: {edge_density:.2f}")
                            
                            # Dilersen klasöre kopyala
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

                            # 1. Yatay veya dikey flip
                            if np.random.rand() > 0.5:
                                aug = cv2.flip(aug, 1)  # yatay
                            if np.random.rand() > 0.7:
                                aug = cv2.flip(aug, 0)  # dikey

                            # 2. Rotation (±90 derece)
                            if np.random.rand() > 0.5:
                                angle = np.random.choice([90, -90])
                                center = (w // 2, h // 2)
                                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                aug = cv2.warpAffine(aug, M, (w, h))

                            # 3. Scaling (zoom in/out + resize)
                            scale_factor = np.random.uniform(0.85, 1.15)
                            scaled = cv2.resize(aug, None, fx=scale_factor, fy=scale_factor)
                            # Ortalanarak crop/pad
                            scaled = cv2.resize(scaled, (w, h))
                            aug = scaled

                            # 5. Border ekle (çerçeve)
                            if np.random.rand() > 0.8:
                                aug = cv2.copyMakeBorder(
                                    aug, 10, 10, 10, 10, borderType=cv2.BORDER_REFLECT
                                )
                                aug = cv2.resize(aug, (w, h))
                            cv2.imwrite(image_path, aug)

                            # Yeni dosya adı
                            new_filename = f"{os.path.splitext(image_file)[0]}_aug{i}.jpg"
                            new_path = os.path.join(class_path, new_filename)
                            cv2.imwrite(new_path, aug)
        print("✔ Veri çoğaltma (flip, rotate, scale, blur, border) tamamlandı.")


dp = DataPreprocessor(folder_dir)
dp.resize_images()
dp.histogram_equalization()
dp.normalize_images()
dp.detect_low_quality_images()
dp.augment_images(num_augmented=2)