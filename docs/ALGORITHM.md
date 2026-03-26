# Understanding the Speculative Decoding Algorithm

Language models are notoriously bound by memory bandwidth constraints; executing autoregressive generation requires shuttling massive parameter weights from High Bandwidth Memory (HBM) into compute cores for every discrete token generated.

**Prolepsis** mitigates this overhead through Speculative Decoding. This paradigm shifts the fundamental bottleneck from sequential memory latency onto the highly parallelized computation capabilities intrinsic to modern GPUs.

## 1. The Draft and Verify Loop

The architecture requires two models that adhere to the exact same tokenizer vocabulary:

1. **Drafting (Fast)**: A smaller, highly efficient architecture ($M_d$) autoregressively generates a sequence of $\gamma$ (gamma) future tokens: $[d_1, d_2, ..., d_\gamma]$. 
2. **Verification (Parallel)**: This proposed sequence is passed down to the target model ($M_t$). Instead of evaluating context recursively, the target model scores the probabilities across all $\gamma$ tokens concurrently via a single forward pass.
3. **Rejection Sampling**: Finally, the predicted probabilities from both models are compared. Tokens are accepted wherever the draft model successfully mirrors the target distribution, and processing seamlessly halts at the first point of unacceptable divergence.

## 2. Modified Rejection Sampling

A defining characteristic of speculative decoding is that it mathematically guarantees the final output perfectly matches the target model's true statistical distribution. This zero-degradation property is achieved using a specialized variant of rejection sampling.

For each drafted token $d_i$:
1. Let $P_d(d_i)$ be the probability assigned by the draft model.
2. Let $P_t(d_i)$ be the probability assigned by the target model.

Whether to **accept** the token $d_i$ is determined by the ratio:

$$ P_{accept} = \min\left(1, \frac{P_t(d_i)}{P_d(d_i)}\right) $$

### 2.1 The Bonus Token & Residual Recovery

If $P_t(d_i) \ge P_d(d_i)$, the target model implicitly agrees with the draft step, and the token is safely accepted with 100% certainty.

If $P_t(d_i) < P_d(d_i)$, a random sampling operation evaluates against $P_{accept}$. If the token is ultimately **rejected**, the acceptance loop immediately halts, and the remainder of the drafted sequence is discarded.

However, a rejected token represents a highly useful statistical gap: measuring exactly *how* the draft model was wrong. The difference between the probabilities is leveraged to sample a mathematically sound replacement token. This dynamically generated replacement is called the **Bonus Token**, and it is drawn exclusively from a renormalized *residual distribution*:

$$ P_{res}(x) = \frac{\max(0, P_t(x) - P_d(x))}{\sum_{x \in V} \max(0, P_t(x) - P_d(x))} $$

Consequently, every verification step is structurally guaranteed to yield at least 1 verified outcome (the bonus token). In the optimal scenario where all $\gamma$ drafted tokens are correct, the engine produces $\gamma$ tokens plus one bonus token drawn from the marginal distribution at the sequence's edge!

## 3. Working with Top-K and Top-P

A large majority of applications employ `top_k` or `top_p` filtering to enforce generation constraints. To preserve statistical integrity throughout rejection sampling, it is strictly required that identical nuclei filters are applied to the raw logits of **both** the draft and target models *prior* to evaluating $P_{accept}$. Prolepsis inherently manages this alignment, enabling standard truncation temperatures without jeopardizing operational math.
