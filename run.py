#!/usr/bin/env python
# This script allows you to run the txm processor without installing the package

import sys
import os

# Add the parent directory to the Python path so txm_processor can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now import from txm_processor
from txm_processor.cli import main

if __name__ == "__main__":
    main()
