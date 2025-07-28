#!/usr/bin/env python3
"""
Evaluation script for InstructDiffusion model on different tasks.

This script evaluates the model performance on each task in isolation using:
- PSNR (Peak Signal-to-Noise Ratio) - via torchmetrics
- SSIM (Structural Similarity Index) - via torchmetrics  
- LPIPS (Learned Perceptual Image Patch Similarity) - via torchmetrics
- FovVideoVDP (Foveated Video Difference Predictor) - via original library

Usage:
    python test_model_evaluation.py --data_path ./data/perceptual_dataset --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt
"""

# Suppress warnings first, before any imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*mat_struct.*")
warnings.filterwarnings("ignore", message=".*scipy.io.matlab.*")

import os
import sys
import math
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn as nn
from torch import autocast
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from einops import rearrange
import einops

import k_diffusion as K
from omegaconf import OmegaConf
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

# Add paths for imports
sys.path.append("./stable_diffusion")
sys.path.append("./")

from stable_diffusion.ldm.util import instantiate_from_config
from dataset.low_level.lowlevel_perceptual import PerceptualDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CFGDenoiser(nn.Module):
    """Classifier-Free Guidance Denoiser for InstructDiffusion."""
    
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
                image_cfg_scale * (out_cond - out_txt_cond)


class TorchMetrics:
    """Collection of torchmetrics-based evaluation metrics."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize torchmetrics
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        
        # Try to initialize LPIPS
        try:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
            self.lpips_available = True
            logger.info("TorchMetrics LPIPS initialized successfully")
        except Exception as e:
            logger.warning(f"TorchMetrics LPIPS not available: {e}")
            self.lpips_available = False
        
        logger.info("TorchMetrics initialized successfully")
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR using torchmetrics."""
        # Ensure tensors have batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return self.psnr(pred, target).item()
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM using torchmetrics."""
        # Ensure tensors have batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return self.ssim(pred, target).item()
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS using torchmetrics."""
        if not self.lpips_available:
            return float('nan')
        
        # Ensure tensors have batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        return self.lpips(pred, target).item()


class FovVideoVDPMetric:
    """FovVideoVDP (Foveated Video Difference Predictor) metric."""
    
    def __init__(self):
        try:
            # Suppress scipy warnings specifically for pyfvvdp
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message=".*mat_struct.*")
            warnings.filterwarnings("ignore", message=".*scipy.io.matlab.*")
            
            import pyfvvdp
            self.fvvdp = pyfvvdp.fvvdp()
            self.available = True
            logger.info("FovVideoVDP metric initialized successfully")
        except ImportError:
            logger.warning("FovVideoVDP not available. Install with: pip install pyfvvdp")
            self.available = False
    
    def compute(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute FovVideoVDP score between two images.
        
        Args:
            img1: First image as numpy array [H, W, C] in range [0, 255]
            img2: Second image as numpy array [H, W, C] in range [0, 255]
            pixels_per_degree: Viewing distance parameter (unused in new implementation)
        
        Returns:
            FovVideoVDP score as float
        """
        if not self.available:
            return float('nan')
        
        try:
            return self.calculate_pyfvvdp(img1, img2)
        except Exception as e:
            logger.warning(f"FovVideoVDP computation failed: {e}")
            return float('nan')
    
    def calculate_pyfvvdp(self, img, ref):
        """
        Calculate FovVideoVDP using the provided function.
        
        Args:
            img: First image as numpy array [H, W, C] in range [0, 255]
            ref: Reference image as numpy array [H, W, C] in range [0, 255]
        
        Returns:
            FovVideoVDP score as float
        """
        import pyfvvdp
        
        # Convert from [H, W, C] to [C, H, W] and normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        ref = torch.from_numpy(ref).permute(2, 0, 1).float() / 255.0
        
        # if the image is in range of -1 to 1, convert it to 0 to 1
        if img.min() < 0:
            img = (img + 1) / 2
            ref = (ref + 1) / 2
            
        fv = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='threshold')
        fvvdp = fv.predict(img, ref, dim_order="CHW")[0].item()
        return fvvdp


class ModelEvaluator:
    """Evaluates InstructDiffusion model on different tasks."""
    
    def __init__(self, config_path: str, ckpt_path: str, device: str = 'cuda'):
        self.device = device
        self.config = OmegaConf.load(config_path)
        self.model = self.load_model(ckpt_path)
        
        # Initialize model components
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])
        
        # Initialize metrics
        self.torch_metrics = TorchMetrics(device)
        self.fvvdp_metric = FovVideoVDPMetric()
        
        logger.info("Model evaluator initialized successfully")
    
    def load_model(self, ckpt_path: str):
        """Load the InstructDiffusion model from checkpoint."""
        model = instantiate_from_config(self.config.model)
        
        logger.info(f"Loading model from {ckpt_path}")
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        if 'state_dict' in pl_sd:
            pl_sd = pl_sd['state_dict']
        
        m, u = model.load_state_dict(pl_sd, strict=False)
        logger.info(f"Model loaded. Missing: {len(m)}, Unexpected: {len(u)}")
        
        model.eval().to(self.device)
        return model
    
    def preprocess_image(self, image: Image.Image, resolution: int = 512) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            resolution: Target resolution
        
        Returns:
            Preprocessed image tensor and original size
        """
        original_size = image.size
        width, height = image.size
        
        # Calculate resize factor
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width_resize = int((width * factor) // 64) * 64
        height_resize = int((height * factor) // 64) * 64
        
        # Resize image
        image = ImageOps.fit(image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = 2 * torch.tensor(np.array(image)).float() / 255 - 1
        image_tensor = rearrange(image_tensor, "h w c -> 1 c h w").to(self.device)
        
        return image_tensor, original_size
    
    def postprocess_image(self, tensor: torch.Tensor, original_size: Tuple[int, int]) -> Image.Image:
        """
        Postprocess model output to PIL Image.
        
        Args:
            tensor: Model output tensor
            original_size: Original image size (width, height)
        
        Returns:
            PIL Image
        """
        # Denormalize
        x = torch.clamp((tensor + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        
        # Convert to PIL
        image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        
        # Resize back to original size
        image = ImageOps.fit(image, original_size, method=Image.Resampling.LANCZOS)
        
        return image
    
    def generate_edited_image(self, input_image: Image.Image, prompt: str, 
                            cfg_text: float = 5.0, cfg_image: float = 1.25, 
                            steps: int = 20, seed: Optional[int] = None) -> Image.Image:
        """
        Generate edited image using InstructDiffusion.
        
        Args:
            input_image: Input PIL Image
            prompt: Text prompt for editing
            cfg_text: Text CFG scale
            cfg_image: Image CFG scale
            steps: Number of denoising steps
            seed: Random seed for reproducibility
        
        Returns:
            Edited PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Preprocess input
        input_tensor, original_size = self.preprocess_image(input_image)
        
        with torch.no_grad(), autocast("cuda"):
            # Prepare conditioning
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([prompt])]
            cond["c_concat"] = [self.model.encode_first_stage(input_tensor).mode()]
            
            # Prepare unconditioning
            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
            
            # Get sampling sigmas
            sigmas = self.model_wrap.get_sigmas(steps)
            
            # Prepare extra args for sampling
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": cfg_text,
                "image_cfg_scale": cfg_image,
            }
            
            # Generate noise and sample
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            
            # Decode to image
            x = self.model.decode_first_stage(z)
        
        # Postprocess
        edited_image = self.postprocess_image(x, original_size)
        
        return edited_image
    
    def compute_metrics(self, pred_image: Image.Image, target_image: Image.Image) -> Dict[str, float]:
        """
        Compute all evaluation metrics between predicted and target images.
        
        Args:
            pred_image: Predicted/generated image
            target_image: Ground truth target image
        
        Returns:
            Dictionary of metric scores
        """
        # Convert to tensors for torchmetrics (range [0, 1])
        pred_tensor = TF.to_tensor(pred_image).to(self.device)
        target_tensor = TF.to_tensor(target_image).to(self.device)
        
        # Compute PSNR, SSIM, and LPIPS using torchmetrics
        psnr = self.torch_metrics.compute_psnr(pred_tensor, target_tensor)
        ssim = self.torch_metrics.compute_ssim(pred_tensor, target_tensor)
        lpips_score = self.torch_metrics.compute_lpips(pred_tensor, target_tensor)
        
        # Convert to numpy arrays for FovVideoVDP (range [0, 255])
        pred_np = np.array(pred_image)
        target_np = np.array(target_image)
        
        # Compute FovVideoVDP
        fvvdp_score = self.fvvdp_metric.compute(pred_np, target_np)
        
        return {
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips_score,
            'FovVideoVDP': fvvdp_score
        }
    
    def evaluate_task(self, task: str, data_path: str, num_samples: int = 50, steps: int = 20, 
                     force_intensity: int = 0, seed: int = 42) -> Dict[str, float]:
        """
        Evaluate model on a specific task.
        
        Args:
            task: Task name (e.g., 'denoise', 'chromo-denoise')
            data_path: Path to dataset
            num_samples: Number of samples to evaluate
            force_intensity: Intensity level (0=none, 1=very mild, 2=mild, 3=extreme)
            seed: Random seed
        
        Returns:
            Dictionary of average metric scores
        """
        logger.info(f"Evaluating task: {task} with {num_samples} samples")
        
        # Create dataset for this task
        try:
            dataset = PerceptualDataset(
                path=data_path,
                split="train",
                size=256,
                task=task,
                force_intensity=force_intensity,
                instruct=False
            )
        except Exception as e:
            logger.error(f"Failed to create dataset for task {task}: {e}")
            return {}
        
        if len(dataset) == 0:
            logger.warning(f"No samples found for task {task}")
            return {}
        
        # Limit number of samples
        num_samples = min(num_samples, len(dataset))
        
        metrics_list = []
        
        for i in range(num_samples):
            try:
                # Get sample from dataset
                sample = dataset[i]
                
                # Extract data
                input_tensor = sample['edit']['c_concat']  # Input image
                target_tensor = sample['edited']  # Target image
                prompt = sample['edit']['c_crossattn']  # Text prompt
                
                # Convert tensors to PIL Images
                input_image = self.tensor_to_pil(input_tensor)
                target_image = self.tensor_to_pil(target_tensor)
                
                # Generate edited image
                pred_image = self.generate_edited_image(
                    input_image, prompt, steps=steps, seed=seed + i
                )
                
                # visualize images
                # Create concatenated visualization: input | predicted | target
                width, height = input_image.size
                concat_image = Image.new('RGB', (width * 3, height))
                concat_image.paste(input_image, (0, 0))
                concat_image.paste(pred_image, (width, 0))
                concat_image.paste(target_image, (width * 2, 0))
                concat_image.save(f"visualization.png")
                
                # Compute metrics
                metrics = self.compute_metrics(pred_image, target_image)
                metrics_list.append(metrics)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{num_samples} samples for task {task}")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i} for task {task}: {e}")
                continue
        
        if not metrics_list:
            logger.error(f"No valid metrics computed for task {task}")
            return {}
        
        # Compute average metrics
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
            else:
                avg_metrics[key] = float('nan')
                avg_metrics[f"{key}_std"] = float('nan')
        
        logger.info(f"Task {task} evaluation completed. PSNR: {avg_metrics.get('PSNR', 'N/A'):.2f}, "
                   f"SSIM: {avg_metrics.get('SSIM', 'N/A'):.4f}, LPIPS: {avg_metrics.get('LPIPS', 'N/A'):.4f}")
        
        return avg_metrics
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        tensor = (tensor * 255).byte()
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        return Image.fromarray(tensor.cpu().numpy())
    
    def evaluate_all_tasks(self, data_path: str, num_samples: int = 50, steps: int = 20,
                          output_csv: str = "evaluation_results.csv") -> pd.DataFrame:
        """Evaluate model on all available tasks."""
        logger.info("Starting evaluation on all tasks")
        
        # Get available tasks
        try:
            temp_dataset = PerceptualDataset(path=data_path, split="train", size=256)
            available_tasks = temp_dataset.get_available_tasks()
            logger.info(f"Found {len(available_tasks)} available tasks: {available_tasks}")
        except Exception as e:
            logger.error(f"Failed to get available tasks: {e}")
            return pd.DataFrame()
        
        results = []
        
        for task in available_tasks:
            # Evaluate with different intensity levels
            for intensity in [0]:
                intensity_name = {0: "none", 1: "very_mild", 2: "mild", 3: "extreme"}[intensity]
                
                logger.info(f"Evaluating task: {task}, intensity: {intensity_name}")
                
                metrics = self.evaluate_task(
                    task=task,
                    data_path=data_path,
                    num_samples=num_samples,
                    steps=steps,
                    force_intensity=intensity
                )
                
                if metrics:
                    result = {
                        'task': task,
                        'intensity': intensity_name,
                        **metrics
                    }
                    results.append(result)
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")
            
            # Print summary
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            print(df.groupby('task')[['PSNR', 'SSIM', 'LPIPS', 'FovVideoVDP']].mean().round(4))
            print("="*80)
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate InstructDiffusion model on different tasks")
    parser.add_argument("--data_path", required=True, help="Path to the perceptual dataset")
    parser.add_argument("--config", default="configs/instruct_diffusion.yaml", help="Path to model config")
    parser.add_argument("--ckpt", default="checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt", help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate per task")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--output_csv", default="evaluation_results.csv", help="Output CSV file for results")
    parser.add_argument("--task", default=None, help="Evaluate specific task only")
    parser.add_argument("--intensity", type=int, default=0, help="Force intensity level (0-3)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        return
    
    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(args.config, args.ckpt, args.device)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation
    if args.task:
        # Evaluate single task
        logger.info(f"Evaluating single task: {args.task}")
        metrics = evaluator.evaluate_task(
            task=args.task,
            data_path=args.data_path,
            num_samples=args.num_samples,
            steps=args.steps,
            force_intensity=args.intensity,
            seed=args.seed
        )
        
        print(f"\nResults for task '{args.task}' with intensity {args.intensity}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    else:
        # Evaluate all tasks
        df = evaluator.evaluate_all_tasks(
            data_path=args.data_path,
            num_samples=args.num_samples,
            steps=args.steps,
            output_csv=args.output_csv
        )
        
        if df.empty:
            logger.error("No evaluation results generated")
        else:
            logger.info(f"Evaluation completed. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
