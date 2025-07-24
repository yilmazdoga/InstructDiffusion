# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Chen Li (edward82@stu.xjtu.edu.cn)
# --------------------------------------------------------

import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import cv2
from PIL import Image
import torchvision


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class PerceptualDataset(Dataset):
    def __init__(self, path, split="train", size=256, interpolation="pil_lanczos", mildness_prob=0.2, execrate_prob=0.2, sample_weight=1.0, instruct=False):
        super(PerceptualDataset, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(path, 'A', split)))
        tar_files = sorted(os.listdir(os.path.join(path, 'B', split)))

        self.inp_filenames = [os.path.join(path, 'A', split, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(path, 'B', split, x) for x in tar_files if is_image_file(x)]

        self.size = size
        self.sample_weight = sample_weight
        self.instruct = instruct
        self.mildness_prob = mildness_prob
        self.execrate_prob = execrate_prob
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.interpolation = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": Image.NEAREST,
            "pil_bilinear": Image.BILINEAR,
            "pil_bicubic": Image.BICUBIC,
            "pil_box": Image.BOX,
            "pil_hamming": Image.HAMMING,
            "pil_lanczos": Image.LANCZOS,
        }[interpolation]

        self.task2natural = {'chromo': 'apply chromostereopsis',
                             'denoise': 'apply denoising',
                             'metamer': 'apply metameric foveation',
                             'quantize': 'apply pixel depth enhancement',
                             'chromo-denoise': 'apply chromostereopsis and denoising',
                             'denoise-metamer': 'apply denoising and metameric foveation',
                             'quantize-metamer': 'apply pixel depth enhancement and metameric foveation',
                             'quantize-chromo': 'apply pixel depth enhancement and chromostereopsis',
                             'denoise-metamer-quantize-chromo': 'apply denoising, apply pixel depth enhancement, apply metameric foveation and apply chromostereopsis',
                             }
        self.mildness_modifiers = ['mildly', 'slightly', 'lightly', 'very slightly', 'very lightly', 'barely']
        self.very_mildness_modifiers = ['very slightly', 'very lightly', 'barely']
        self.execrate_modifiers = ['highly', 'strongly', 'extremely']
        
        print(f"PerceptualDataset has {len(self)} samples!!")

    def __len__(self):
        return int(self.sizex * self.sample_weight)

    def __getitem__(self, index):
        if self.sample_weight >= 1:
            index_ = index % self.sizex
        else:
            index_ = int(index / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        width, height = inp_img.size
        tar_width, tar_height = tar_img.size
        assert tar_width == width and tar_height == height, "Input and target image mismatch"
        aspect_ratio = float(width) / float(height)
        if width < height:  
            new_width = self.size  
            new_height = int(self.size  / aspect_ratio)  
        else:  
            new_height = self.size   
            new_width = int(self.size * aspect_ratio) 
        inp_img = inp_img.resize((new_width, new_height), self.interpolation)
        tar_img = tar_img.resize((new_width, new_height), self.interpolation)

        inp_img = np.array(inp_img).astype(np.float32).transpose(2, 0, 1)
        inp_img_tensor = torch.tensor((inp_img / 127.5 - 1.0).astype(np.float32))
        tar_img = np.array(tar_img).astype(np.float32).transpose(2, 0, 1)
        tar_img_tensor = torch.tensor((tar_img / 127.5 - 1.0).astype(np.float32))
        
        image_0 = inp_img_tensor
        image_1 = tar_img_tensor

        # Extract filename without extension and use it as prompt
        filename = os.path.basename(inp_path)
        prompt = os.path.splitext(filename)[0]  # Remove file extension
        
        # Extract task from filename (remove image ID)
        # Example: "0001_chromo-denoise.png" -> "chromo-denoise"
        parts = prompt.split('_')
        if len(parts) > 1:
            task = '_'.join(parts[1:])  # Everything after the first underscore
        else:
            task = prompt  # Fallback if no underscore
        
        # Generate natural language prompt using task2natural mapping
        if task in self.task2natural:
            natural_prompt = self.task2natural[task]
        else:
            # Fallback: use the original filename-based approach
            natural_prompt = task.replace('_', ' ').replace('-', ' ')
        
        # Apply mildness with probability mildness_prob
        if random.random() < self.mildness_prob:
            # Add mildness modifier and interpolate images
            mildness_modifier = random.choice(self.mildness_modifiers)
            modified_prompt = f"{mildness_modifier} " + natural_prompt
            
            # Adjust interpolation factor based on mildness level
            if mildness_modifier in self.very_mildness_modifiers:
                # Very mild modifications: interpolation factor between 0.2 and 0.4
                interpolation_factor = random.uniform(0.2, 0.4)
            else:
                # Regular mild modifications: interpolation factor between 0.4 and 0.6
                interpolation_factor = random.uniform(0.4, 0.6)
            
            final_image = image_0 * (1 - interpolation_factor) + image_1 * interpolation_factor
        else:
            # No mildness applied - check for execrate operation
            if random.random() < self.execrate_prob:
                # Add execrate modifier and multiply image by 1.2
                execrate_modifier = random.choice(self.execrate_modifiers)
                modified_prompt = f"{execrate_modifier} " + natural_prompt
                
                # Multiply target image by 1.2 to make effect stronger
                # Clamp to prevent values going beyond valid range
                final_image = torch.clamp(image_1 * 1.2, -1.0, 1.0)
            else:
                # Use original target image and prompt without modification
                modified_prompt = natural_prompt
                final_image = image_1
        
        if self.instruct:
            prompt = "Image Enhancement: " + modified_prompt
        else:
            prompt = modified_prompt
        
        return dict(edited=final_image, edit=dict(c_concat=image_0, c_crossattn=prompt))
    
if __name__ == "__main__":
    # Example usage
    dataset_path = "./data/perceptual_dataset"
    
    # Create dataset instance
    dataset = PerceptualDataset(
        path=dataset_path,
        split="train",
        size=256,
        instruct=False
    )
    
    # Get one sample
    sample = dataset[0]
    
    print(f"Sample 0:")
    print(f"  Input shape: {sample['edit']['c_concat'].shape}")
    print(f"  Output shape: {sample['edited'].shape}")
    print(f"  Prompt: '{sample['edit']['c_crossattn']}'")
    print(f"  Input range: [{sample['edit']['c_concat'].min():.3f}, {sample['edit']['c_concat'].max():.3f}]")
    print(f"  Output range: [{sample['edited'].min():.3f}, {sample['edited'].max():.3f}]")
    
    # Convert tensors back to PIL Images and save
    def tensor_to_pil(tensor):
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        # Clamp to valid range
        tensor = torch.clamp(tensor, 0, 1)
        # Convert to [0, 255] range
        tensor = (tensor * 255).byte()
        # Convert CHW to HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        # Convert to numpy
        np_img = tensor.numpy()
        # Convert to PIL Image
        return Image.fromarray(np_img)
    
    # Save input and output images
    input_img = tensor_to_pil(sample['edit']['c_concat'])
    output_img = tensor_to_pil(sample['edited'])
    
    input_img.save("in.png")
    output_img.save("out.png")
    
    print(f"\nImages saved:")
    print(f"  Input image: in.png")
    print(f"  Output image: out.png")