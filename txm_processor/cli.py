import argparse
import os
import sys
import warnings
from rich.console import Console
from .core import process_volume

# Suppress warnings from skimage
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

def parse_slice_range(value):
    """Parse comma-separated values as integers"""
    try:
        return [int(x) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: {value}. Expected comma-separated numbers.")

def main():
    parser = argparse.ArgumentParser(description="Process TXM volume files with efficient patchification")
    
    parser.add_argument("txm_file", help="Path to the TXM file to process")
    
    parser.add_argument("--from", dest="from_slice", type=parse_slice_range,
                        help="Starting slice for each axis as comma-separated values (xy,yz,xz). E.g., 100,90,200")
    
    parser.add_argument("--to", dest="to_slice", type=parse_slice_range,
                        help="Ending slice for each axis as comma-separated values (xy,yz,xz). E.g., 800,900,1000")
    
    parser.add_argument("--patch-size", type=int, default=224,
                        help="Size of patches (default: 224)")
    
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same as TXM filename)")
    
    parser.add_argument("--no-clahe", action="store_true",
                        help="Disable CLAHE enhancement")
    
    parser.add_argument("--no-histeq", action="store_true",
                        help="Disable histogram equalization")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.txm_file):
        console.print(f"[bold red]Error: File not found: {args.txm_file}")
        sys.exit(1)
    
    # Parse slice ranges
    from_slices = None
    to_slices = None
    
    if args.from_slice:
        from_slices = {}
        if len(args.from_slice) >= 1:
            from_slices['xy'] = args.from_slice[0]
        if len(args.from_slice) >= 2:
            from_slices['yz'] = args.from_slice[1]
        if len(args.from_slice) >= 3:
            from_slices['xz'] = args.from_slice[2]
    
    if args.to_slice:
        to_slices = {}
        if len(args.to_slice) >= 1:
            to_slices['xy'] = args.to_slice[0]
        if len(args.to_slice) >= 2:
            to_slices['yz'] = args.to_slice[1]
        if len(args.to_slice) >= 3:
            to_slices['xz'] = args.to_slice[2]
    
    # Set output directory
    output_dir = args.out_dir or os.path.splitext(os.path.basename(args.txm_file))[0]
    
    # Process the volume
    process_volume(
        args.txm_file,
        from_slice=from_slices,
        to_slice=to_slices,
        patch_size=args.patch_size,
        output_dir=output_dir,
        apply_clahe=not args.no_clahe,
        apply_histeq=not args.no_histeq
    )

if __name__ == "__main__":
    main(
        args.txm_file,
        from_slice=from_slices if from_slices else None,
        to_slice=to_slices if to_slices else None,
        patch_size=args.patch_size,
        output_dir=output_dir,
        apply_clahe=not args.no_clahe,
        apply_histeq=not args.no_histeq
    )

if __name__ == "__main__":
    main()
