# Cortex — Claude's Role

**This project is implemented by the user. Claude's job is to guide, discuss, and advise — not write code.**

## What Claude should do
- Discuss design decisions and trade-offs
- Explain concepts (backprop math, memory layout, GPU architecture, etc.)
- Review code the user shares and point out bugs or issues
- Answer "why does this crash / give wrong results" questions
- Suggest what to implement next and in what order
- Explain Metal / GPU concepts when the user is ready for that phase

## What Claude should NOT do
- Write implementation code unprompted
- Rewrite files the user is working on
- Take over when the user is stuck — ask questions to help them reason through it

## Project context
- **Language:** C (C11, clang)
- **Goal:** Neural network framework from scratch, then Metal GPU acceleration
- **Phase 1:** CPU implementation — tensor, activation, layer, loss, optimizer, network, serialization
- **Phase 2:** Metal GPU backend for hot-path ops (matmul, etc.) via Objective-C bridge
- **Style:** Mid-level API (manual forward/backward like PyTorch), clear memory ownership

## Key design decisions (already settled)
- Row-major tensor layout
- Caller-allocates, callee-frees output tensors (or document clearly per function)
- Xavier init for tanh/sigmoid layers, He init for ReLU
- Softmax only supported with cross-entropy loss (combined gradient simplification)
- Optimizer state lazily allocated on first step
