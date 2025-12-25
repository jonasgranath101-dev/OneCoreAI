# OneCoreAI

![Logo Description](/logo.png)

OneCoreAI is a block-based AI system written in C that allows creating multiple AI "cores" (blocks), each containing modular AI logic with extractable variables.

## Screenshots

![screenshot 1](/.screenshots/Screenshot%202025-12-24%207.50.40%20AM.png)

![screenshot 2](/.screenshots/Screenshot%202025-12-24%207.51.46%20AM.png)

![screenshot 3](/.screenshots/Screenshot%202025-12-24%207.52.16%20AM.png)

![screenshot 4](/.screenshots/Screenshot%202025-12-24%208.39.00%20AM.png)

## Features

- **Multiple AI Cores**: Create and manage multiple independent AI processing units
- **Modular AI Blocks**: AI logic broken into reusable blocks (forward pass, loss calculation, gradient computation, parameter updates)
- **Extractable Variables**: Save and load learned parameters, hyperparameters, and training history
- **Block-Based Architecture**: Clean separation of AI components for easy extension
- **Linear Regression**: Built-in linear regression with gradient descent training

## Architecture

Each AI Core contains:
- **Forward Block**: Prediction calculation (w*x + b)
- **Loss Block**: Mean Squared Error calculation
- **Gradient Block**: Parameter gradient computation
- **Update Block**: Parameter optimization via gradient descent
- **Training Block**: Orchestrates the training process
- **Prediction Block**: Makes predictions using learned parameters
- **Variable Extraction Blocks**: Extract/save/load core variables

## Usage

Compile the program:
```bash
cd .core
gcc -o onecoreai init.c src.c -lm
./onecoreai
```

The demonstration creates 3 AI cores with different learning rates and epochs, trains them on synthetic data (y = 2*x + 1 + noise), and shows prediction accuracy.

## Core Management

- Create cores with different configurations
- Train cores individually or simultaneously
- Extract variables for analysis or persistence
- Ensemble predictions across multiple cores
- Save/load core state to/from files

## File Structure

- `.core/init.c`: Main program and core management
- `.core/src.c`: Additional AI block functions
- `.core/handle.h`: Header with function prototypes and AICore structure
- `.lib/variable.txt`: Variable format documentation
- `.tool/configure.txt`: Configuration storage
- `.tool/.logs/log.txt`: Program diagnostics

## Extensions

The modular block design allows easy addition of:
- Different loss functions
- Advanced optimizers
- Regularization techniques
- Cross-validation
- Model persistence
- Ensemble methods

## Future features.

  ~ Block location.
  ~ Data size on block.
