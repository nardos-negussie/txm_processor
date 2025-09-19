import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
from pathlib import Path
from skimage import exposure
from skimage.util import view_as_blocks
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
import threading
import matplotlib.pyplot as plt
from skimage.io import imsave

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up console for rich output
console = Console()

# Handle dependency imports with better error messages
try:
    import xrmreader
except ImportError as e:
    console.print(f"[bold red]Error: Missing dependency - {e}")
    console.print("[yellow]Please install required dependencies with:")
    console.print("[green]pip install xrmreader")
    console.print("[yellow]If you've already installed them but still get errors, try:")
    console.print("[green]pip install --upgrade tomopy xrmreader")
    raise

def load_txm_file(txm_path, console=console):
    """Load a TXM file using xrmreader with progress indication"""
    with console.status(f"[bold blue]Loading TXM file {os.path.basename(txm_path)}...", spinner="dots"):
        start_time = time.time()
        volume = xrmreader.read_txm(txm_path)
        elapsed = time.time() - start_time
        console.print(f"[green]Loaded TXM file in {elapsed:.2f} seconds. Shape: {volume.shape}")
    return volume

def extract_slice(volume, axis, slice_idx):
    """Extract a slice from the volume along a specific axis"""
    if axis == 'xy':
        return volume[slice_idx, :, :]
    elif axis == 'yz':
        return volume[:, slice_idx, :]
    elif axis == 'xz':
        return volume[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be one of ['xy', 'yz', 'xz']")

def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    return exposure.equalize_adapthist(image)

def apply_histeq(image):
    """Apply histogram equalization"""
    return exposure.equalize_hist(image)

def calculate_padding(image_shape, patch_size):
    """Calculate padding needed to ensure the image is divisible by patch_size"""
    h, w = image_shape
    pad_h = (0, patch_size - h % patch_size) if h % patch_size != 0 else (0, 0)
    pad_w = (0, patch_size - w % patch_size) if w % patch_size != 0 else (0, 0)
    return pad_h, pad_w

def patchify(image, patch_size, verbose=False):
    """Split image into patches of size patch_size x patch_size"""
    h, w = image.shape
    
    # Calculate necessary padding
    pad_h, pad_w = calculate_padding(image.shape, patch_size)
    
    # Apply padding if needed
    if pad_h[1] > 0 or pad_w[1] > 0:
        padded_img = np.pad(image, ((0, pad_h[1]), (0, pad_w[1])), mode='reflect')
    else:
        padded_img = image
    
    # Create patches manually to ensure correctness
    grid_h = (h + pad_h[1]) // patch_size
    grid_w = (w + pad_w[1]) // patch_size
    
    patches = []
    
    # Only print debug info if verbose is enabled
    if verbose:
        console.print(f"[yellow]Original image shape: {image.shape}")
        console.print(f"[yellow]Padded image shape: {padded_img.shape}")
        console.print(f"[yellow]Grid dimensions: {grid_h}x{grid_w}")
        console.print(f"[yellow]Patch size: {patch_size}")
    
    # Extract each patch
    for i in range(grid_h):
        for j in range(grid_w):
            start_h = i * patch_size
            start_w = j * patch_size
            end_h = start_h + patch_size
            end_w = start_w + patch_size
            
            # Extract the patch
            patch = padded_img[start_h:end_h, start_w:end_w]
            
            # Verify patch shape
            if patch.shape != (patch_size, patch_size):
                if verbose:
                    console.print(f"[bold red]Unexpected patch shape: {patch.shape} at position ({i},{j})")
                continue
                
            patches.append(patch)
    
    # Verify we have the correct number of patches
    expected_patches = grid_h * grid_w
    if len(patches) != expected_patches and verbose:
        console.print(f"[bold red]Warning: Expected {expected_patches} patches, got {len(patches)}")
    
    # Convert to numpy array for consistency
    patches = np.array(patches)
    
    # Debug visualization is removed to reduce output verbosity
    
    return patches, (grid_h, grid_w), ((0, pad_h[1]), (0, pad_w[1]))

def process_slice(params):
    """Process a single slice (CPU-intensive part)"""
    volume, axis, idx, patch_size, apply_clahe_flag, apply_histeq_flag = params
    
    # Extract the slice
    slice_img = extract_slice(volume, axis, idx)
    
    # Check if slice contains valid data
    if np.isnan(slice_img).any() or np.isinf(slice_img).any():
        console.print(f"[bold red]Warning: Slice {axis}:{idx} contains invalid values. Fixing...")
        slice_img = np.nan_to_num(slice_img, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalize slice to 0-1 range for equalization operations
    slice_min, slice_max = slice_img.min(), slice_img.max()
    if slice_max > slice_min:
        slice_img = (slice_img - slice_min) / (slice_max - slice_min)
    
    # Apply enhancements if requested (CPU-intensive)
    if apply_clahe_flag:
        try:
            slice_img = apply_clahe(slice_img)
        except Exception as e:
            console.print(f"[bold red]Error applying CLAHE to {axis}:{idx}: {e}")
    
    if apply_histeq_flag:
        try:
            slice_img = apply_histeq(slice_img)
        except Exception as e:
            console.print(f"[bold red]Error applying histogram equalization to {axis}:{idx}: {e}")
    
    # Patchify the slice (CPU-intensive)
    patches, grid_shape, padding = patchify(slice_img, patch_size)
    
    return idx, axis, patches, grid_shape, padding, slice_img.shape

def save_patches(params):
    """Save processed patches to disk (IO-intensive part)"""
    idx, axis, patches, grid_shape, padding, original_shape, output_dir, patch_size = params
    
    # Get base filename without extension
    base_filename = os.path.basename(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata for reconstruction (if needed)
    metadata = {
        'axis': axis,
        'slice_idx': idx,
        'original_shape': original_shape,
        'grid_shape': grid_shape,
        'padding': padding,
        'patch_size': patch_size
    }
    
    # Save each patch with the naming convention
    for i, patch in enumerate(patches):
        # Format: FNAME_VIEW_S_SLICENUM_P_PATCHNUM.png
        patch_filename = f"{base_filename}_{axis}_S_{idx}_P_{i}.png"
        patch_path = os.path.join(output_dir, patch_filename)
        
        # Normalize patch for saving as PNG
        patch_normalized = (patch * 255).astype(np.uint8) if patch.max() <= 1.0 else patch.astype(np.uint8)
        imsave(patch_path, patch_normalized)
    
    return idx, axis

def process_volume(txm_path, from_slice=None, to_slice=None, patch_size=224, 
                  output_dir=None, apply_clahe=True, apply_histeq=False):
    """Process a volume file by extracting slices, applying enhancements, and patchifying"""
    # Track total processing time
    start_time = time.time()
    
    console.print(f"[bold blue]Processing file: {txm_path}")
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = Path(txm_path).stem
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load volume
    volume = load_txm_file(txm_path)
    
    # Determine slice ranges for each axis
    depth, height, width = volume.shape
    
    # Visualize one slice from each axis to check if reading works properly
    console.print("[yellow]Visualizing example slices from each axis for verification...")
    
    # Get middle slices for visualization
    mid_xy = depth // 2
    mid_yz = width // 2
    mid_xz = height // 2
    
    # Extract middle slices
    slice_xy = extract_slice(volume, 'xy', mid_xy)
    slice_yz = extract_slice(volume, 'yz', mid_yz)
    slice_xz = extract_slice(volume, 'xz', mid_xz)
    
    # Save the slices for inspection
    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    
    # Normalize for better visualization
    def normalize_for_display(img):
        # Scale to 0-255 range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            normalized = ((img - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
        return normalized
    
    # Save preview images
    xy_path = os.path.join(preview_dir, f"preview_xy_slice_{mid_xy}.png")
    yz_path = os.path.join(preview_dir, f"preview_yz_slice_{mid_yz}.png")
    xz_path = os.path.join(preview_dir, f"preview_xz_slice_{mid_xz}.png")
    
    imsave(xy_path, normalize_for_display(slice_xy))
    imsave(yz_path, normalize_for_display(slice_yz))
    imsave(xz_path, normalize_for_display(slice_xz))
    
    console.print(f"[green]Preview slices saved to {preview_dir}")
    console.print(f"[green]XY slice shape: {slice_xy.shape}")
    console.print(f"[green]YZ slice shape: {slice_yz.shape}")
    console.print(f"[green]XZ slice shape: {slice_xz.shape}")
    
    # Create enhanced versions of slices for comparison
    # Normalize slices to 0-1 range for enhancement operations
    def normalize_for_enhancement(img):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(img)
    
    # Create enhanced versions
    slice_xy_norm = normalize_for_enhancement(slice_xy)
    slice_xy_clahe = exposure.equalize_adapthist(slice_xy_norm)
    slice_xy_histeq = exposure.equalize_hist(slice_xy_norm)
    
    # Also create a combined figure with original and enhanced views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: Original views of different axes
    axes[0, 0].imshow(slice_xy, cmap='gray')
    axes[0, 0].set_title(f'XY Slice {mid_xy}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(slice_yz, cmap='gray')
    axes[0, 1].set_title(f'YZ Slice {mid_yz}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(slice_xz, cmap='gray')
    axes[0, 2].set_title(f'XZ Slice {mid_xz}')
    axes[0, 2].axis('off')
    
    # Second row: Enhancement comparison for XY slice
    axes[1, 0].imshow(slice_xy_norm, cmap='gray')
    axes[1, 0].set_title('Original (Normalized)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(slice_xy_clahe, cmap='gray')
    axes[1, 1].set_title('CLAHE Enhanced')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(slice_xy_histeq, cmap='gray')
    axes[1, 2].set_title('Histogram Equalized')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(preview_dir, "preview_all_views_with_enhancement.png"), dpi=150)
    plt.close()
    
    # Save individual enhanced images
    imsave(os.path.join(preview_dir, "xy_clahe.png"), (slice_xy_clahe * 255).astype(np.uint8))
    imsave(os.path.join(preview_dir, "xy_histeq.png"), (slice_xy_histeq * 255).astype(np.uint8))
    
    console.print(f"[bold green]Preview images saved to {preview_dir}")
    console.print("[yellow]Please check the preview images and then continue processing")
    console.print("[yellow]Enhancement comparison is shown in the preview_all_views_with_enhancement.png file")
    
    # Ask user if they want to continue with processing
    user_input = input("Check preview images. Continue with processing? (y/n): ").strip().lower()
    if user_input != 'y':
        console.print("[bold red]Processing aborted by user")
        return output_dir
    
    # Default ranges use full volume dimensions
    slice_ranges = {
        'xy': (0, depth),
        'yz': (0, width),
        'xz': (0, height)
    }
    
    # Override with provided ranges if any
    if from_slice and 'xy' in from_slice:
        slice_ranges['xy'] = (from_slice['xy'], to_slice.get('xy', depth))
    if from_slice and 'yz' in from_slice:
        slice_ranges['yz'] = (from_slice['yz'], to_slice.get('yz', width))
    if from_slice and 'xz' in from_slice:
        slice_ranges['xz'] = (from_slice['xz'], to_slice.get('xz', height))
    
    # Create processing tasks for all slices
    processing_tasks = []
    
    for axis, (start, end) in slice_ranges.items():
        for idx in range(start, end):
            processing_tasks.append((volume, axis, idx, patch_size, apply_clahe, apply_histeq))
    
    # Setup progress tracking
    total_slices = len(processing_tasks)
    console.print(f"[yellow]Total slices to process: {total_slices}")
    
    # Set number of workers for threading
    num_workers = min(32, os.cpu_count() * 4)  # Use more threads since we're only using threading now
    console.print(f"[blue]Using {num_workers} threads for all operations")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100  # Process 100 slices at a time
    total_batches = (total_slices + batch_size - 1) // batch_size
    
    console.print(f"[yellow]Processing in {total_batches} batches of {batch_size} slices each")
    
    # Process first batch with debug output, rest without
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_slices)
        
        batch_tasks = processing_tasks[start_idx:end_idx]
        processed_results = []
        
        console.print(f"[yellow]Processing batch {batch_num + 1}/{total_batches} (slices {start_idx} to {end_idx-1})")
        
        # Set first batch to verbose, others to quiet
        is_verbose = (batch_num == 0)
        
        # Modify the batch tasks to include the verbose flag if needed
        if is_verbose and batch_num == 0:
            # First batch: process first slice with verbosity
            modified_tasks = []
            for i, task in enumerate(batch_tasks):
                if i == 0:
                    # For the first task, we'll handle it separately to show debug info
                    volume, axis, idx, ps, ac, ah = task
                    # Process this task with the original function but with verbose output
                    try:
                        slice_img = extract_slice(volume, axis, idx)
                        if np.isnan(slice_img).any() or np.isinf(slice_img).any():
                            slice_img = np.nan_to_num(slice_img, nan=0.0, posinf=1.0, neginf=0.0)
                        slice_min, slice_max = slice_img.min(), slice_img.max()
                        if slice_max > slice_min:
                            slice_img = (slice_img - slice_min) / (slice_max - slice_min)
                        if ac:
                            slice_img = apply_clahe(slice_img)
                        if ah:
                            slice_img = apply_histeq(slice_img)
                        patches, grid_shape, padding = patchify(slice_img, ps, verbose=True)
                        console.print("[green]First slice processed successfully - continuing with reduced verbosity")
                    except Exception as e:
                        console.print(f"[bold red]Error processing first slice: {e}")
                modified_tasks.append(task)
            batch_tasks = modified_tasks
        
        # Setup progress bars for both processing and saving
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            processing_task = progress.add_task(f"[cyan]Processing batch {batch_num + 1}/{total_batches}...", 
                                              total=len(batch_tasks))
            
            # Use ThreadPoolExecutor instead of multiprocessing Pool
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all processing tasks
                future_to_task = {executor.submit(process_slice, task): task for task in batch_tasks}
                
                # As they complete, collect results
                for future in future_to_task:
                    try:
                        idx, axis, patches, grid_shape, padding, original_shape = future.result()
                        processed_results.append((idx, axis, patches, grid_shape, padding, original_shape))
                        progress.update(processing_task, advance=1)
                    except Exception as e:
                        console.print(f"[bold red]Error processing slice: {e}")
            
            # Prepare saving tasks
            saving_tasks = [(idx, axis, patches, grid_shape, padding, original_shape, output_dir, patch_size) 
                            for idx, axis, patches, grid_shape, padding, original_shape in processed_results]
            
            saving_task = progress.add_task(f"[green]Saving batch {batch_num + 1}/{total_batches}...", 
                                          total=len(saving_tasks))
            
            # Use ThreadPoolExecutor for saving as well
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all saving tasks
                future_to_save = {executor.submit(save_patches, task): task for task in saving_tasks}
                
                # As they complete, update progress
                for future in future_to_save:
                    try:
                        future.result()
                        progress.update(saving_task, advance=1)
                    except Exception as e:
                        console.print(f"[bold red]Error saving patches: {e}")
        
        # Force garbage collection to free memory
        processed_results.clear()
        import gc
        gc.collect()
    
    # Calculate and display total processing time
    end_time = time.time()
    total_seconds = end_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    console.print(f"[bold green]Processing complete! Output saved to: {output_dir}")
    console.print(f"[bold blue]Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return output_dir
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    console.print(f"[bold green]Processing complete! Output saved to: {output_dir}")
    console.print(f"[bold blue]Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return output_dir
