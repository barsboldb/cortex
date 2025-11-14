# Neural Network Framework in C - Implementation Plan

## Project Structure

```
cortex/
├── include/
│   ├── tensor.h          # Core tensor/matrix operations
│   ├── layer.h           # Layer abstractions
│   ├── activation.h      # Activation functions
│   ├── loss.h            # Loss functions
│   ├── optimizer.h       # Optimizers (SGD, Adam, etc.)
│   ├── network.h         # Network building & training
│   └── serialization.h   # Save/load models
├── src/
│   ├── tensor.c
│   ├── layer.c
│   ├── activation.c
│   ├── loss.c
│   ├── optimizer.c
│   ├── network.c
│   └── serialization.c
├── examples/
│   ├── xor.c            # XOR classifier example
│   └── mnist.c          # MNIST digit recognition
├── tests/
│   └── test_*.c         # Unit tests
├── Makefile
└── README.md
```

## Core Components

### 1. Tensor/Matrix Module (`tensor.h/c`)
- Dynamic matrix structure with shape tracking
- Basic operations: creation, destruction, copy
- Math operations: matrix multiply, add, subtract, scalar ops
- Element-wise operations
- Random initialization (Xavier, He)

### 2. Activation Functions (`activation.h/c`)
- Sigmoid, ReLU, Leaky ReLU, tanh, Softmax
- Forward and backward (derivative) functions
- Function pointer interface for flexibility

### 3. Layer Abstractions (`layer.h/c`)
- Dense/Fully-connected layer structure
- Forward pass (manual control like PyTorch)
- Backward pass (gradient computation)
- Layer parameters and gradients storage

### 4. Loss Functions (`loss.h/c`)
- MSE (Mean Squared Error)
- Cross-Entropy
- Forward and backward functions

### 5. Optimizers (`optimizer.h/c`)
- SGD (with optional momentum)
- Adam optimizer
- RMSprop
- Update function to apply gradients to parameters

### 6. Network Builder (`network.h/c`)
- Network structure holding layers
- Add/remove layers
- Forward pass through all layers
- Backward pass through all layers
- Training loop utilities

### 7. Serialization (`serialization.h/c`)
- Save model weights to binary file
- Load model weights from file
- Save/load optimizer state

## Implementation Steps

1. **Set up project structure** - Create directories and Makefile
2. **Implement tensor module** - Core matrix operations foundation
3. **Implement activation functions** - All forward/backward functions
4. **Implement layer abstraction** - Dense layer with backprop
5. **Implement loss functions** - MSE and Cross-Entropy
6. **Implement optimizers** - SGD, Adam, RMSprop
7. **Implement network builder** - Connect layers, training loop
8. **Implement serialization** - Save/load functionality
9. **Create XOR example** - Verify framework works
10. **Add documentation** - README with API usage examples

## Key Design Decisions

- **Mid-level API**: Manual forward/backward control for flexibility
- **Memory management**: Clear ownership semantics (caller frees)
- **Extensibility**: Function pointers for activations/optimizers
- **GPU-ready**: Structure designed for future GPU/Neural Engine acceleration
- **Simple dependencies**: Pure C with standard library only
