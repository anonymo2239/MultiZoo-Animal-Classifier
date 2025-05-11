import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

class TorchPreprocessor:
    def _init_(self, input_folder, output_folder, image_size=(224, 224)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_size = image_size

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(image_size)

        self.augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.4)
        ]

        os.makedirs(self.output_folder, exist_ok=True)

    def process_and_save(self):
        for root, _, files in os.walk(self.input_folder):
            for fname in tqdm(files, desc="Preprocessing"):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(input_path, self.input_folder)
                    rel_folder = os.path.dirname(rel_path)
                    output_subfolder = os.path.join(self.output_folder, rel_folder)
                    os.makedirs(output_subfolder, exist_ok=True)

                    try:
                        image = Image.open(input_path).convert("RGB")
                    except Exception as e:
                        print(f"✗ {fname} açılamadı: {e}")
                        continue

                    image = self.resize(image)
                    tensor_img = self.to_tensor(image).cuda()

                    all_augmented = [tensor_img]

                    for aug in self.augmentations:
                        pil_img = self.to_pil(tensor_img.cpu())
                        aug_img = aug(pil_img)
                        aug_tensor = self.to_tensor(aug_img).cuda()
                        all_augmented.append(aug_tensor)

                    base_name, ext = os.path.splitext(fname)
                    for i, t in enumerate(all_augmented):
                        final_pil = self.to_pil(t.cpu())
                        out_name = f"{base_name}_aug{i}{ext}"
                        save_path = os.path.join(output_subfolder, out_name)
                        final_pil.save(save_path)

                    print(f"✓ {rel_path} işlendi ve {len(all_augmented)} varyasyon kaydedildi.")

input_dir = "C:/train"
output_dir = "C:/train_aug"

pre = TorchPreprocessor(input_dir, output_dir)
pre.process_and_save()