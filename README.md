# TXM Processor

A Python package for efficient processing of TXM (3D volume) files with real-time progress tracking, multi-threading for I/O operations, and multi-processing for CPU-intensive tasks.

## Features

- Extract XY, YZ, and XZ views from 3D volumes
- Apply CLAHE and histogram equalization to slices
- Patchify slices with automatic padding for reconstruction guarantee
- Multi-threaded I/O operations for improved performance
- Multi-processed computation for CPU-intensive tasks
- Rich terminal output with progress tracking

## Installation

### From source

```bash
python -m pip install  git+https://github.com/nardos-negussie/txm_processor.git
```


To fix error during import xrmreader , i.e, the `dxchange.writer` issue, please follow these steps:

```bash
python -m pip install git+https://github.com/data-exchange/dxchange.git

```

### Dependencies

- numpy
- scikit-image


### Dependencies

- numpy
- scikit-image
git clone https://github.com/data-exchange/dxchange.git
cd dxchange
python -m pip install -e .

this fixed it. could you please mention the error (during import xrmreader the dxchange.writer issue)

### Dependencies

- numpy
- scikit-image
- rich
- xrmreader
- matplotlib

## Usage

### Command Line Interface

```bash
txm-process your_volume.txm [OPTIONS]
```

#### Options

- `--from VALUES` - Starting slices for each axis (xy,yz,xz) as comma-separated values (e.g., `--from 100,90,200`)
- `--to VALUES` - Ending slices for each axis (xy,yz,xz) as comma-separated values (e.g., `--to 800,900,1000`)
- `--patch-size SIZE` - Size of patches (default: 224)
- `--out-dir DIR` - Output directory (default: same as TXM filename)
- `--no-clahe` - Disable CLAHE enhancement
- `--no-histeq` - Disable histogram equalization

#### Examples

Process all slices with default settings:
```bash
txm-process sample.txm
```

Process specific slice ranges:
```bash
txm-process sample.txm --from 100,90,200 --to 800,900,1000
```

or 

```bash
python ./run.py beans_Tomo_AreaA_RECON.txm --from 80,200,250 --to 780,700,750 --no-histeq --patch-size 224 --no-clahe
```

Disable CLAHE and use a custom patch size:
```bash
txm-process sample.txm --no-clahe --patch-size 512
```

### Python API

```python
from txm_processor.core import process_volume

# Process a TXM file
output_dir = process_volume(
    "path/to/volume.txm",
    from_slice={'xy': 100, 'yz': 90, 'xz': 200},
    to_slice={'xy': 800, 'yz': 900, 'xz': 1000},
    patch_size=224,
    output_dir="output",
    apply_clahe=True,
    apply_histeq=False
)
```

## Output Structure

The output directory will have the following structure:

