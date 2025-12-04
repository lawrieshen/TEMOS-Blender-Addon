# TEMOS-Blender-Addon

A Blender add-on that integrates the TEMOS text-to-motion model directly into Blender, enabling users to generate and visualize 3D character motion from text prompts inside their 3D scenes.

## Overview

The TEMOS Blender Add-on provides a seamless pipeline for generating human motion directly from natural language descriptions within Blender. By connecting Blender’s interface to the TEMOS text-to-motion generation model, the add-on lets you:

- Enter a text prompt (e.g. *“a person walks forward and waves”*).
- Generate a corresponding 3D motion clip.
- Visualize and inspect the motion directly in the Blender viewport.

This workflow removes the need for manual export/import steps and keeps the entire ideation–to–animation process inside Blender.



https://github.com/user-attachments/assets/7e97da51-6744-43cb-8e4d-ac5aaf5a618c



https://github.com/user-attachments/assets/695530d4-6be8-432f-be35-fb2a8d9d7ca0



### Academic Context

This project contains the code and experiments developed for an undergraduate honours thesis at the University of Sydney titled:

**“Integrating Deep Learning and 3D Animation into Physical Rehabilitation”**

The add-on is used as part of a broader research pipeline that explores how deep learning–driven motion generation and 3D animation can support physical therapy and rehabilitation workflows.

### Acknowledgment

This add-on is built on, and directly interfaces with, the official PyTorch implementation of:

> **TEMOS: Generating diverse human motions from textual descriptions**  
> Petrovich, Mathis; Black, Michael J.; Varol, Gül.  
> European Conference on Computer Vision (ECCV), 2022.

Official project page and code:  
https://mathis.petrovich.fr/temos/

```bibtex
@inproceedings{petrovich22temos,
  title     = {{TEMOS}: Generating diverse human motions from textual descriptions},
  author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {European Conference on Computer Vision ({ECCV})},
  year      = {2022}
}
```

## Getting Started

### Prerequisites

* **Blender** 4.3+ (or a later LTS version). This code is tested on Blender 4.3.
* A working installation of the **TEMOS** PyTorch implementation and its dependencies (see the official repository for setup instructions:
  [https://mathis.petrovich.fr/temos/](https://mathis.petrovich.fr/temos/) or follow the instructions provided later).
* This repository (the **TEMOS-Blender-Addon**) cloned.
* Disk space of ~1.5 GB for downloading the pretrained models.

> ⚠️ Note: The add-on assumes you have a Python/conda environment where TEMOS can be executed. For more information, please consult the installation instructions later.

### Installation

1. **Prepare dependencies with conda**

   * Create a conda/miniconda virtual environment in your terminal by running:

   ```bash
   conda env create -f configs/temos_env.yml
   conda activate temos
   ```

   If you encounter any issues at this step, please consult the installation instructions in the official TEMOS repository: [https://github.com/Mathux/TEMOS](https://github.com/Mathux/TEMOS)

2. **Download the add-on and pretrained models**

   * Clone this repository into any directory you prefer.
   * With the `temos` environment activated, `cd` into the project directory and run:

   ```bash
   # install gdown (optional if you already have it)
   # pip install --user gdown

   cd models/TEMOS
   bash prepare/download_pretrained_models.sh
   ```

3. **Install in Blender**

   * With the `temos` environment still activated, launch Blender from the terminal so that the correct dependencies are available:

   ```bash
   # For macOS
   /Applications/Blender.app/Contents/MacOS/Blender

   # For Linux (if Blender is in your PATH)
   /usr/bin/blender
   ```

   * Locate the Python installation used by Blender by running:

   ```bash
   <BLENDER_EXECUTABLE> --background --python-expr "import sys; import os; print('\nThe path to the installation of python of blender can be:'); print('\n'.join(['- '+x.replace('/lib/python', '/bin/python') for x in sys.path if 'python' in (file:=os.path.split(x)[-1]) and not file.endswith('.zip')]))"
   ```

   I will refer to this path as `/path/to/blender/python`.

   * Install `pip` into Blender’s Python:

   ```bash
   /path/to/blender/python -m ensurepip --upgrade
   ```

   * Install the required packages into Blender’s Python environment:

   ```bash
   /path/to/blender/python -m pip install --user numpy
   /path/to/blender/python -m pip install --user matplotlib
   /path/to/blender/python -m pip install --user hydra-core --upgrade
   /path/to/blender/python -m pip install --user hydra_colorlog --upgrade
   /path/to/blender/python -m pip install --user moviepy
   /path/to/blender/python -m pip install --user shortuuid
   ```

   If you encounter any issues at this step, please consult the original TEMOS code implementation.

   * In Blender, go to:
     `Edit` → `Preferences…` → `Add-ons`.
   * Click **Install…**, select the `__init__.py` file for this add-on, and confirm.
   * Search for **“Motion Generator”** in the add-ons list and tick the checkbox to enable it.

### Basic Usage

1. **Prepare your scene**

   * Open or create a Blender scene.
   * Add a character/armature compatible with your motion output (or a placeholder rig for testing).

2. **Open the Motion Generator panel**

   * In the 3D Viewport, open the **N-panel** (press `N`).
   * Navigate to the **“Motion Generator”** tab (the tab created by this add-on).

3. **Generate motion**

   * Enter a **text prompt** describing the motion (e.g. *“a person slowly raises their left arm and waves”*).
   * Set the desired length of the motion sequence to be generated.
   * Click **Preview Motion”** to run a lightweight inference step. You can inspect the onion-skin image in the UI window as a preview.
   * Click **“Generate Motion”** to create the full motion visualization in the Blender viewport and apply it to your scene.
