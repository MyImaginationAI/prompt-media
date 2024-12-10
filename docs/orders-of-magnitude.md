# Orders of Magnitude (OOM) in AI-Generated Media

## Introduction

Orders of Magnitude (OOM) is a concept used to introduce controlled variations into prompts, particularly in the context of AI-generated media. It allows for the creation of dynamic and diverse content while maintaining coherence across different prompts.

## Why We Need OOM

OOM is essential in AI-generated content creation for several reasons:

1. **Diversity Within Prompts**: Each prompt can have unique combinations of elements while maintaining thematic consistency.
2. **Controlled Randomization**: Random elements are used only once within each prompt, ensuring variety without repetition.
3. **Coherent Storytelling**: Different prompts can tell a cohesive story with varying elements.

## Types of OOM

### Sequential (`:sequential`)

Elements cycle in a specific order across prompts. For example:

```yaml
variations:
  hand_gestures:
    type: "sequential"
    values: ["thoughtful hands", "gesturing hands", "pointing hands"]
```

This will cycle through options in order:
- First prompt: "thoughtful hands"
- Second prompt: "gesturing hands"
- Third prompt: "pointing hands"
- Fourth prompt: back to "thoughtful hands"

The sequence continues cyclically through all prompts, ensuring a predictable progression of elements.

### Random (`:random`)

Elements are chosen randomly for each prompt, but each option is used only once within that prompt. For example:

```yaml
variations:
  lighting:
    type: "random"
    values: ["warm studio lights", "cool studio lights", "neutral lighting"]
```

For each prompt:
- Randomly selects one unused option
- Won't repeat until all options have been used
- Resets available options for the next prompt

### Static (`:static`)

A single element remains constant across all prompts. For example:

```yaml
variations:
  camera_angle:
    type: "static"
    values: ["wide studio shot"]
```

This will use "wide studio shot" in every prompt, providing consistency.

## Example Output

Given these variations:
```yaml
variations:
  time:
    type: "sequential"
    values: ["morning", "afternoon", "evening", "night"]
  style:
    type: "random"
    values: ["realistic", "artistic", "abstract"]
```

And this base prompt:
```yaml
prompts:
  - "A serene mountain landscape at sunset"
```

The prompts will be generated like:
1. "A serene mountain landscape at sunset, morning, realistic"
2. "A serene mountain landscape at sunset, afternoon, artistic"
3. "A serene mountain landscape at sunset, evening, abstract"
4. "A serene mountain landscape at sunset, night, abstract"

## Application Example

Here's how OOMs work in practice:
