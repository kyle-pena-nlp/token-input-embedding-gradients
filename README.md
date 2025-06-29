# Input Space Gradients

This repository contains tools for analyzing and visualizing gradient-based manipulations in the input embedding space of transformer models.

## Project Setup

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- Git

### Getting Started

After cloning the repository, follow these steps to set up the project:

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install project dependencies**:
   ```bash
   cd input_space_gradients
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```
   This will automatically:
   - Clear Jupyter notebook outputs before commits
   - Remove trailing whitespace
   - Fix end of files
   - Check YAML syntax

### Working with the Project

#### Adding Dependencies

- **Add a new dependency**:
  ```bash
  poetry add package-name
  ```

- **Add a development dependency**:
  ```bash
  poetry add --group dev package-name
  ```

#### Managing the Environment

- **View the dependency tree**:
  ```bash
  poetry show --tree
  ```

- **Export dependencies** to requirements.txt format:
  ```bash
  poetry export -f requirements.txt -o requirements.txt
  ```

- **Update dependencies**:
  ```bash
  poetry update
  ```

#### Running Tests

```bash
poetry run pytest
```

### Jupyter Notebooks

This project uses `nbstripout` to automatically clear notebook outputs before committing. This keeps version control history clean and focused on code changes rather than outputs.

To manually clean notebook outputs:
```bash
pre-commit run nbstripout --all-files
```

To temporarily bypass the pre-commit hooks when committing:
```bash
git commit --no-verify
```

## Project Structure

- `bert_*.py` - BERT model utilities and visualizations
- `llama_*.py` - LLaMA model utilities and computations
- `compute_batch_*_gradients.py` - Gradient computation for respective models
