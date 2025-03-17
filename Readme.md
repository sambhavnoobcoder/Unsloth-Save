# LLM Optimization Challenges

This repository contains solutions to several advanced challenges in optimizing Large Language Models (LLMs) for training and inference. All solutions were developed and tested on Kaggle notebooks with 2x Tesla T4 GPUs, demonstrating that sophisticated LLM techniques can be implemented even with limited computational resources.

## Challenges Overview

### Challenge 1: Quantization Optimization
Implementing efficient quantization and dequantization techniques for NF4 format to reduce memory footprint while maintaining model performance.

**Solution Highlights:**
- Custom dequantization function for NF4 weights
- Memory-efficient implementation compatible with PyTorch
- Optimized for inference on consumer-grade GPUs

### Challenge 2: QLoRA with FSDP2 Multi-GPU Training
Making Quantized Low-Rank Adaptation (QLoRA) work with Fully Sharded Data Parallel v2 (FSDP2) across multiple GPUs.

**Solution Highlights:**
- Successfully implemented QLoRA fine-tuning with FSDP2 on 2x Tesla T4 GPUs
- Integrated BitsAndBytes 4-bit quantization with distributed training
- Implemented proper gradient accumulation and mixed precision training
- Achieved equivalent loss to single-GPU training
- Demonstrated with Llama 3.1 8B model

### Challenge 3: Optimized Transformer Compilation
Enhancing transformer models with PyTorch compilation techniques while handling dynamic sequence lengths.

**Solution Highlights:**
- Implemented flexible attention with dynamic sequence length support
- Successfully integrated BitsAndBytes with torch.compile
- Compiled attention mechanisms without excessive recompilation
- Optimized MLP modules, loss functions, and layer normalization
- Enabled Triton MatMul autotuning for maximum performance

### Challenge 5: Memory-Efficient Backpropagation
Implementing memory-efficient backpropagation for LLMs to reduce VRAM usage during training.

**Solution Highlights:**
- Created a custom torch.autograd.Function for memory-efficient linear operations
- Implemented batched computation of intermediate tensors to reduce memory usage
- Successfully reduced VRAM usage by 2-5% during training
- Demonstrated with both Cross Entropy and Focal Loss functions
- Maintained gradient correctness while reducing memory footprint

## Implementation Details

All challenges were implemented and tested on Kaggle notebooks with 2x Tesla T4 GPUs, demonstrating that advanced LLM optimization techniques can be achieved even with limited computational resources.

The solutions prioritize:
- Memory efficiency
- Computational performance
- Compatibility with existing frameworks
- Numerical stability
- Ease of integration

## Kaggle Notebooks

- [Challenge 1: Quantization Optimization](https://www.kaggle.com/code/sambhavdixit/unsloth-challenge-1-final-submission)
- [Challenge 2: QLoRA with FSDP2](https://www.kaggle.com/code/sambhavdixit/unsloth-challenge-2-final-submission)
- [Challenge 3: Optimized Transformer Compilation](https://www.kaggle.com/code/sambhavdixit/unsloth-challenge-3-final-submission)
- [Challenge 5: Memory-Efficient Backpropagation](https://www.kaggle.com/code/sambhavdixit/unsloth-challenge-5-final-submission)

## Medium Articles

Detailed explanations of the techniques used in these challenges are available in the following Medium articles:

- [Challenge 1](https://medium.com/@indosambhav/unsloth-challenge-1-convert-nf4-to-triton-e6571899cf21)
- [Challenge 2](https://medium.com/your-username/article2)
- [Challenge 3](https://medium.com/@indosambhav/unsloth-challenge-3-submission-a460d9b29f20)
- [Challenge 5](https://medium.com/@indosambhav/unsloth-challenge-5-memory-effecient-backprop-3e7f74b29d99)

## Getting Started

Each challenge directory contains the complete code and documentation needed to reproduce the results. The code is designed to work with PyTorch and Hugging Face Transformers.


### Running the Code

Each challenge can be run independently. See the individual challenge directories for specific instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the PyTorch and Hugging Face teams for their excellent libraries
- Special thanks to Kaggle for providing the GPU resources needed for this project
