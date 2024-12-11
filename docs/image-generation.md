# Image Generation Workflow

## Overview
This document describes the image generation workflow in the Prompt Media project, including how images are generated, saved, and organized.

## File Organization
Generated images are saved in a datetime-based directory structure:
```
collections/images/YYYY/MM/DD/HHMM/
```

For example:
```
collections/images/2024/12/11/1458/
```

## File Naming Convention
Images are saved with a unique filename that includes:
- A sequential number (global count)
- The seed used for generation

Format: `{count:03d}_seed_{seed}.png`

Example: `001_seed_12345.png`

### Global Count System
The global count ensures that each generated image gets a unique number, preventing file overwrites:
- Count starts at 0 for each generation session
- Increments for each image generated, regardless of the prompt or variation
- Persists across all prompts and their OOM (Orders of Magnitude) variations
- Format uses three digits with leading zeros (e.g., 001, 002, 099)

## Workflow Process
1. **Session Start**
   - Initialize global count at 0
   - Create datetime-based output directory

2. **For Each Prompt**
   - Generate variations based on OOM configurations
   - For each variation:
     - Generate image(s)
     - Save with incrementing global count
     - Update global count for next generation

3. **File Saving**
   - Copy prompt configuration file to output directory
   - Save images with unique filenames
   - Log saved image paths

## LoRA Integration
The workflow supports dynamic integration of LoRA (Low-Rank Adaptation) models:

### Configuration
LoRAs are configured in the prompt media file:
```yaml
loras:  # Optional list of LoRA configurations
  - name: "model1.safetensors"
    strength_model: 0.5  # Model strength (0.0 to 1.0)
    strength_clip: 0.5   # CLIP strength (0.0 to 1.0)
  - name: "model2.safetensors"
    strength_model: 0.6
    strength_clip: 0.6
```

### Workflow Behavior
1. **No LoRAs Specified**
   - Base model (checkpoint) is used directly
   - CLIP and KSampler nodes connect to checkpoint outputs

2. **Single LoRA**
   - LoRA is applied to the base model
   - CLIP and KSampler use the LoRA's outputs

3. **Multiple LoRAs**
   - LoRAs are chained in sequence
   - Each LoRA builds upon the previous one's transformations
   - Final LoRA's outputs are used for CLIP and KSampler
   - Order matters: first LoRA is applied to base model, second to first's output, etc.

### Example Chain
For a configuration with three LoRAs:
```
Checkpoint → LoRA1 → LoRA2 → LoRA3 → KSampler/CLIP
```
- Each LoRA preserves its individual strength settings
- Transformations are cumulative through the chain

## Example
If you have 3 prompts with 2 OOM variations each:
```
001_seed_12345.png  # First prompt, first variation
002_seed_67890.png  # First prompt, second variation
003_seed_11111.png  # Second prompt, first variation
004_seed_22222.png  # Second prompt, second variation
005_seed_33333.png  # Third prompt, first variation
006_seed_44444.png  # Third prompt, second variation
```

## Related Documentation
- [Orders of Magnitude (OOM)](orders-of-magnitude.md) - Details on how variations are generated
- [Image Composer](tools/image-composer.md) - Information about the image composition tool
