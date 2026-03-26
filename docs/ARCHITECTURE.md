# Prolepsis Architecture Overview

To ensure code readability, testing viability, and maintainability, the fundamental mathematics of speculative decoding have been cleanly isolated from the complex internal state requirements of Hugging Face transformer models.

The framework is structurally divided into the following isolated systems:

## 1. Core Domains

### 1.1 The Orchestrator: `SpeculativeDecoder`
Located natively in `core/speculative_engine.py`, this class functions as the primary orchestration pipeline. It provides the `.generate()` interface end-users expect without leaking internal caching complexity. 

Functional execution is rigidly delegated to dedicated helpers:
- **`draft_tokens()`** $\rightarrow$ executed by the `DraftModelWrapper`
- **`verify_tokens()`** $\rightarrow$ executed by the `TargetModelWrapper`
- **`run_rejection_sampling()`** $\rightarrow$ executed by the `RejectionSampler`

### 1.2 Managing the State: `DualKVCacheManager`
The most prominent failure vector in custom sequential decoding revolves around maintaining synchronized Key-Value (KV) caches. When the target model rejects a token, the active KV cache layers must be dynamically "rewound" precisely to the execution point immediately prior to the failure.

The `DualKVCacheManager` is constructed entirely for this purpose:
- It tracks the logical and discrete generation lengths of the two respective models perfectly.
- It calculates the absolute `sync_point` identifying where truncation must operate post-rejection.
- It precisely increments context pointers mapping token advancement rules (tracking edge-case insertions like the calculated "bonus token" offset).

### 1.3 Keeping Models Tidy: `ModelWrappers`
Instead of force-mutating standard Hugging Face classes, foundation models are safely wrapped inside `DraftModelWrapper` and `TargetModelWrapper` interfaces.
- Both inherit behavior from a shared `_KVCacheMixin`, enabling native abilities to securely truncate and offset their internal contextual mappings (`past_key_values`) autonomously.
- The wrappers are dynamically flexible, seamlessly bridging traditional decoupled tuple-based caches and emergent `DynamicCache` unified structures.

## 2. A Note on Performance

Inside `core/rejection_sampler.py`, the token probability evaluations are implemented leveraging purely decoupled PyTorch tensor manipulations. This approach evaluates the entire sequence of $\gamma$ drafted tokens synchronously across the model's entire vocabulary space without iterating through slow CPU loops.

### Optional Triton Integration
For production use cases requiring maximal throughput on massive inference hardware, an optional custom verify implementation is provided inside `kernels/verify_kernel.py`. This specialized instruction set relies on OpenAI's Triton compiler to aggressively fuse the acceptance condition branch logic and RNG probability comparisons directly onto the GPU hardware streaming multiprocessors.
