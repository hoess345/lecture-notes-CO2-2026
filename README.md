# LectureNotes

Repository for lecture notes, scripts and notebooks for the CO2 course 2026.

## Structure

Each exercise is its own self-contained folder with all necessary notebooks, utility scripts, and data files.

```
LectureNotes/
├── exercise_1/          # SVM Example
├── exercise_2/          # Keras Intro
└── pyproject.toml       # Environment configuration
```

## Environment

The Python environment is managed with [uv](https://docs.astral.sh/uv/), but usage is optional. You can also use pip, poetry, or any other package manager of your choice.

**With uv:**

```bash
uv sync
```

## Exercises

| Exercise        | Topic                      | Description                                                                                                                                   |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exercise 1**  | SVM Fundamentals         | Introduction to Keras, activation functions (Sigmoid, ReLU, Leaky ReLU), dropout, batch normalization, skip connections, Keras Functional API |
| **Exercise 2**  | Keras Fundamentals         | Introduction to Keras, activation functions (Sigmoid, ReLU, Leaky ReLU), dropout, batch normalization, skip connections, Keras Functional API |

## Requirements

- Keras 3.x with TensorFlow backend (or PyTorch/JAX)
- See `pyproject.toml` for full dependency list
