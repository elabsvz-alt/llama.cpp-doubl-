# Scientific Investigation: Mathematical Feasibility of Large-Scale LLM Inference on Resource-Constrained CPU Architectures

## 1. Abstract & Mathematical Formalization

### Abstract
This investigation explores the mathematical and architectural feasibility of executing Large Language Models (LLMs) with a memory footprint of 5–10 GB on a hardware environment limited to 4 GB of System RAM and a legacy CPU architecture (Intel Ivy Bridge i5-3470). The core scientific challenge lies in the violation of the *Memory-to-Model Capacity Ratio*, where the physical volatile memory is insufficient to house the static weight tensors. Standard inference paradigms rely on loading the entire model into RAM to satisfy the low-latency requirements of autoregressive decoding. In this constrained environment, we formalize a mathematical mutation that treats weight matrices not as static entities, but as dynamic streams or decomposed low-rank approximations to bypass the 4 GB RAM barrier and the ~10.6 GB/s bandwidth bottleneck of single-channel DDR3-1333 memory.

### Mathematical Formalization of the Baseline
In a standard Transformer-based LLM, the primary computational load is centered on the Matrix-Vector Multiplication ($MV$):

$$Y = WX$$

Where:
- $W \in \mathbb{R}^{m \times n}$ represents the weight matrix of a linear layer (e.g., in the MLP or Attention blocks).
- $X \in \mathbb{R}^{n \times 1}$ is the input activation vector (hidden state).
- $Y \in \mathbb{R}^{m \times 1}$ is the resulting activation.

Under standard floating-point operations (e.g., $FP16$), each element $w_{ij}$ consumes 2 bytes. For a 7B parameter model, $W$ occupies ~14 GB. The computation for a single element $y_i$ is defined as:

$$y_i = \sum_{j=1}^{n} w_{ij} x_j$$

In the context of the user-defined hardware (Ivy Bridge i5-3470), this operation is strictly **Memory Bandwidth Bound**. The Arithmetic Intensity ($AI$) of a matrix-vector product is:

$$AI = \frac{\text{Floating Point Operations}}{\text{Bytes Transferred}} = \frac{2mn}{2mn + 2n} \approx 1 \text{ FLOP/Byte}$$

Given the peak theoretical bandwidth of 10.6 GB/s and a CPU throughput of ~102.4 GFLOPS (AVX), the system is underutilized by a factor of ~10x due to data starvation. The challenge is exacerbated when the model exceeds physical RAM, requiring the inclusion of storage I/O latency into the mathematical formalization of the compute cost.

## 2. Linear Algebra Bottleneck Analysis

### 2.1 Memory Bandwidth vs. Compute Throughput
The Intel i5-3470 (Ivy Bridge) features AVX vector extensions, capable of processing 256-bit registers. At 3.2 GHz, the theoretical peak throughput for single-precision floating point ($FP32$) is:
$$\text{Peak GFLOPS} = 4 \text{ cores} \times 8 \text{ flops/cycle (AVX)} \times 3.2 \text{ GHz} = 102.4 \text{ GFLOPS}$$
For half-precision ($FP16$), although Ivy Bridge supports $F16C$ for conversion, it does not have native $FP16$ arithmetic units; thus, throughput remains bound by the $FP32$ rate.

In contrast, the memory subsystem consists of a single 4GB DDR3-1333 module. The theoretical peak bandwidth ($B_m$) is:
$$B_m = 1333 \text{ MHz} \times 8 \text{ bytes (64-bit bus)} \times 1 \text{ channel} \approx 10.66 \text{ GB/s}$$

For a 7B parameter model in $Q4_0$ quantization (~3.8 GB), a single forward pass requires transferring the entire weight set from RAM to the CPU caches. The time taken for memory transfer $T_m$ is:
$$T_m = \frac{\text{Model Size}}{\text{Bandwidth}} = \frac{3.8 \text{ GB}}{10.66 \text{ GB/s}} \approx 0.356 \text{ seconds per token}$$
This limits the generation speed to approximately **2.8 tokens per second**, assuming ideal cache conditions and zero computational overhead.

### 2.2 The Capacity Gap & Paging Penalty
When the model size reaches 10 GB, it exceeds the 4 GB physical RAM by 250%. Standard BLAS implementations (e.g., OpenBLAS, Intel MKL) assume that the working set resides in the primary address space. Operating system demand paging (swap) introduces a catastrophic bottleneck:
- **SATA SSD Bandwidth:** ~0.5 GB/s.
- **Resulting Latency:** $T_{m\_swap} = \frac{10 \text{ GB}}{0.5 \text{ GB/s}} = 20 \text{ seconds per token}$.

Mathematical mutations must therefore focus on reducing the active working set size or changing the fundamental structure of the linear algebra routine to allow for asynchronous computation while data is "in-flight" from storage.

### 2.3 Cache Hierarchy Constraints
The 6 MB L3 cache is shared across 4 cores. In a standard $Y=WX$ operation, the vector $X$ and the partial results $Y$ must stay in cache, while blocks of $W$ are streamed. With a hidden dimension $h=4096$, the vector $X$ ($FP32$) consumes 16 KB, which easily fits in L1. However, standard matrix-vector multiplication does not exploit temporal locality of the weight matrix $W$ in autoregressive decoding, as each weight is used exactly once per token.

## 3. Algorithmic Mutation & Optimization Proposal

To overcome the architectural limitations, we propose the **Asymmetric Tensor Decomposition with Residual Streaming (ATD-RS)** mutation.

### 3.1 Mathematical Mutation: Low-Rank Residual Decomposition
We mutate the static weight matrix $W$ into a sum of a persistent high-precision low-rank approximation and a transient ultra-low-precision residual:

$$W = U V^T + Q(R)$$

Where:
- $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$ are low-rank factors ($r \ll n, m$) stored in RAM in $FP16$ or $BF16$.
- $R = W - U V^T$ is the residual matrix.
- $Q(\cdot)$ is a non-uniform, block-wise 2-bit quantization operator.

The Matrix-Vector multiplication is reformulated as:
$$Y = U(V^T X) + Q(R)X$$

### 3.2 Alteration of Memory Access Patterns
This mutation splits the computation into two distinct phases with different architectural alignments:

1.  **Phase A (Compute-Bound):** $X' = V^T X$, then $Y_{base} = U X'$.
    - Since $U$ and $V$ are small ($2 \times m \times r$ elements), they are locked in RAM or even the L3 cache.
    - The complexity drops from $O(mn)$ to $O(r(m+n))$.
    - This phase exploits the CPU's high GFLOPS by increasing the arithmetic intensity.

2.  **Phase B (Bandwidth-Bound Streaming):** $Y_{res} = Q(R)X$.
    - $Q(R)$ is stored on storage (SSD) and streamed into a small ring buffer in RAM.
    - Using 2-bit quantization ($Q2\_K$), the 10 GB model residual is compressed to ~2.5 GB, fitting within the remaining 4 GB RAM alongside the OS and KV cache.
    - The operation $Q(R)X$ is executed using SSSE3 bit-parallel logic, where 8 elements are processed per register cycle.

### 3.3 Dynamic Tensor Streaming (DTS)
To hide SSD latency, we propose a look-ahead streaming mechanism. While the CPU calculates the $i$-th layer's $Y = UV^T X$, the $i+1$-th layer's residual $Q(R_{i+1})$ is prefetched from storage into a DMA-aligned buffer. The mathematical execution time $T_{total}$ becomes:
$$T_{total} = \sum_{l=1}^{L} \max(T_{compute}(U_l, V_l, X), T_{fetch}(Q(R_l)))$$
By matching the 2-bit residual size to the DDR3 bandwidth, we theoretically approach 100% hardware efficiency.

## 4. Feasibility Matrix & Comparative Analysis

The following table summarizes the theoretical performance for a 7B parameter model (Baseline size ~14 GB) on the target hardware (4 GB RAM, Ivy Bridge CPU).

| Metric | Baseline CPU (FP16) | Existing CPU Opt. (INT4) | Proposed Mutation (ATD-RS) |
| :--- | :--- | :--- | :--- |
| **Memory Footprint** | ~14 GB | ~3.8 GB | **~2.1 GB** |
| **RAM Occupancy** | > 350% (Swap Death) | ~95% (Marginal) | **~52% (Safe)** |
| **Bandwidth Utilization** | < 2% (Disk Bound) | ~90% (DDR3 Bound) | **~100% (Balanced)** |
| **Computational Complexity**| $2mn$ | $2mn$ | **$2r(m+n) + mn_{bits}$** |
| **Theoretical Tok/s** | 0.05 | 2.8 | **4.5 - 5.0** |

*Note: ATD-RS complexity assumes $r=128$ and $n_{bits}$ represents bitwise operations for the 2-bit residual.*

## 5. Implementation Roadmap & Hardware Alignment

### 5.1 Ivy Bridge Vector Primitives
The proposed mutation utilizes specific primitives of the i5-3470 architecture:
1.  **_mm_maddubs_epi16 (SSSE3):** Due to the lack of AVX2 on Ivy Bridge, integer dot products must utilize 128-bit SSSE3 registers. This instruction performs the dot product of 8-bit unsigned and signed integers, allowing for efficient dequantization-accumulation cycles in the 2-bit residual path.
2.  **F16C Instruction Set:** Utilized for rapid conversion of low-rank factors ($U, V$) from $FP16$ storage to $FP32$ registers for Phase A computation via `_mm256_cvtph_ps`.
3.  **L3 Cache Locking:** The small low-rank factors (estimated at ~500 KB per layer for $r=128$) are sized to fit within the 6 MB L3 cache, effectively eliminating DRAM latency for the persistent part of the model.

### 5.2 Dynamic Residual Buffer (DRB)
To adhere to the 4 GB limit, the implementation must utilize a double-buffering scheme:
- **Buffer 1:** Active residual $Q(R_l)$ being computed upon.
- **Buffer 2:** Look-ahead residual $Q(R_{l+1})$ being fetched from SSD via asynchronous `mmap` or `io_uring`.

### 5.3 Mathematical Mutation Alignment
The mutation from $W \rightarrow \{U, V, Q(R)\}$ ensures that the "heavy" part of the model (the residual) is only accessed sequentially, maximizing the throughput of the DDR3 prefetcher and avoiding random access patterns that would stall the Ivy Bridge pipeline.
