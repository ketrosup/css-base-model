#!/usr/bin/env python
"""Execute the defect detection notebook"""
import subprocess
import sys

try:
    from jupyter_client.kernelspec import KernelSpecManager
    from nbconvert import exporters

    print("Executing notebook: defect_detection_cnn.ipynb")
    print("=" * 60)

    # Use nbconvert to execute the notebook
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    # Read the notebook
    with open('defect_detection_cnn.ipynb') as f:
        nb = nbformat.read(f, as_version=4)

    # Execute it
    ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})

    # Write the executed notebook
    with open('defect_detection_cnn_output.ipynb', 'w') as f:
        nbformat.write(nb, f)

    print("=" * 60)
    print("✓ Notebook executed successfully!")
    print("Output saved to: defect_detection_cnn_output.ipynb")
    sys.exit(0)

except Exception as e:
    print(f"✗ Error executing notebook: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
