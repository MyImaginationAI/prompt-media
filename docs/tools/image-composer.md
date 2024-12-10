# Image Composer Tools

This document describes the image composition tools available in the project for creating GIFs and collages from your generated images.

## Overview

The image composer tools provide two main functionalities:
1. Creating animated GIFs from a sequence of images
2. Creating image collages with customizable layouts

The tools are available through the Makefile and can be used with any directory containing images.

## Usage

### Creating a Collage

```bash
make compose/collage INPUT_DIR=path/to/images/directory
```

Options:
- `BORDER_SIZE`: Horizontal spacing between images in pixels (default: 2)
- `ROW_SPACING`: Vertical spacing between rows in pixels (default: same as BORDER_SIZE)
- `COLLAGE_WIDTH`: Maximum width of the collage in pixels (default: 2400)
- `OUTPUT_COLLAGE`: Custom output path for the collage (default: INPUT_DIR/collage.jpg)

Example with custom settings:
```bash
make compose/collage \
    INPUT_DIR=collections/images/2024/12/10/1308/ \
    BORDER_SIZE=1 \
    ROW_SPACING=1 \
    COLLAGE_WIDTH=3000
```

### Creating a GIF

```bash
make compose/gif INPUT_DIR=path/to/images/directory
```

Options:
- `GIF_DURATION`: Duration for each frame in milliseconds (default: 500)
- `OUTPUT_GIF`: Custom output path for the GIF (default: INPUT_DIR/output.gif)

Example with custom settings:
```bash
make compose/gif \
    INPUT_DIR=collections/images/2024/12/10/1308/ \
    GIF_DURATION=1000 \
    OUTPUT_GIF=my-animation.gif
```

### Creating Both GIF and Collage

To create both outputs at once:

```bash
make compose INPUT_DIR=path/to/images/directory
```

This command accepts all the options mentioned above for both GIF and collage creation.

## Technical Details

### Collage Layout
- Images are arranged in rows of 5 to optimize for wide-format images
- Aspect ratio is preserved based on the first image in the directory
- Minimal spacing between images can be configured using BORDER_SIZE and ROW_SPACING
- The collage width is adjustable via COLLAGE_WIDTH parameter

### GIF Creation
- Maintains original image dimensions
- All frames are resized to match the first image's dimensions
- Supports customizable frame duration
- Creates an infinitely looping animation

## Implementation

The tools are implemented in Python using the Pillow library for image processing. The source code is located at:
- `tools/image_composer.py`: Main implementation
- `Makefile`: Command-line interface and configuration
