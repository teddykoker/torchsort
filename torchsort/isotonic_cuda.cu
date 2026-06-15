//  Copyright 2007-2020 The scikit-learn developers.
//  Copyright 2020 Google LLC.
//  Copyright 2021 Teddy Koker.
//  All rights reserved.
// 
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
// 
//    a. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//    b. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//    c. Neither the name of the Scikit-learn Developers  nor the names of
//       its contributors may be used to endorse or promote products
//       derived from this software without specific prior written
//       permission.
// 
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.


#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

//  Copied from fast-soft-sort (https://bit.ly/3r0gOav) with the following modifications:
//  - replace numpy functions with torch equivalents
//  - re-write in CUDA
//  - return solution in place
//  - added backward pass (vector jacobian product)

//  Copied from scikit-learn with the following modifications:
//  - use decreasing constraints by default,
//  - do not return solution in place, rather save in array `sol`,
//  - avoid some needless multiplications.
// namespace {

// Numerically stable log-add-exp
template <typename scalar_t>
__device__ __forceinline__ scalar_t log_add_exp(scalar_t x, scalar_t y) {
    scalar_t larger = max(x, y);
    scalar_t smaller = min(x, y);
    return larger + log1p(exp(smaller - larger));
}

// Returns partition corresponding to solution. Expects sizes to be zeros
template <typename scalar_t>
__device__ void partition(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> solution, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sizes, 
    int n, 
    int b) {

    const scalar_t eps = 1.0e-9;
    int tail = 1;

    if (n > 0) {
        sizes[b][0] = 1;
    }

    for (int i = 1; i < n; i++) {
        if (std::abs(solution[b][i] - solution[b][i - 1]) > eps) {
            tail += 1; 
        }
        sizes[b][tail - 1] += 1;
    }
}


template <typename scalar_t>
__global__ void isotonic_l2_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sums,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> target,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> c,
    int n,
    int batch) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    // target describes a list of blocks.  at any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        c[b][i] = 1.0;
        sol[b][i] = s[b][i];
        sums[b][i] = s[b][i];
        target[b][i] = i;
    }

    int i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        if (k == n) {
            break;
        }
        if (sol[b][i] > sol[b][k]) {
            i = k;
            continue;
        }
        auto sum_y = sums[b][i];
        auto sum_c = c[b][i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[b][k];
            sum_y += sums[b][k];
            sum_c += c[b][k];
            k = target[b][k] + 1;
            if ((k == n) || (prev_y > sol[b][k])) {
                // Non-singleton increasing subsequence is finished,
                // update first entry.
                sol[b][i] = sum_y / sum_c;
                sums[b][i] = sum_y;
                c[b][i] = sum_c;
                target[b][i] = k - 1;
                target[b][k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can.  This makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[b][i - 1];
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        for (int j = i + 1; j < k; j++) {
            sol[b][j] = sol[b][i];
        }
        i = k;
    }
}

template <typename scalar_t>
__global__ void isotonic_kl_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> lse_y_,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> lse_w_,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> target,
    int n,
    int batch) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    // target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        sol[b][i] = y[b][i] - w[b][i];
        lse_y_[b][i] = y[b][i];
        lse_w_[b][i] = w[b][i];
        target[b][i] = i;
    }

    int i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        if (k == n) {
            break;
        }
        if (sol[b][i] > sol[b][k]) {
            i = k;
            continue;
        }
        auto lse_y = lse_y_[b][i];
        auto lse_w = lse_w_[b][i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[b][k];
            lse_y = log_add_exp(lse_y, lse_y_[b][k]);
            lse_w = log_add_exp(lse_w, lse_w_[b][k]);
            k = target[b][k] + 1;
            if ((k == n) || (prev_y > sol[b][k])) {
                // Non-singleton increasing subsequence is finished,
                // update first entry.
                sol[b][i] = lse_y - lse_w;
                lse_y_[b][i] = lse_y;
                lse_w_[b][i] = lse_w;
                target[b][i] = k - 1;
                target[b][k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can.  This makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[b][i - 1];
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        for (int j = i + 1; j < k; j++) {
            sol[b][j] = sol[b][i];
        }
        i = k;
    }
}


template <typename scalar_t>
__global__ void isotonic_l2_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s, // not used
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ret,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sizes,
    int n,
    int batch) {

    int end;
    scalar_t sum;
    scalar_t val;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    int start = 0;
    partition(sol, sizes, n, b);
    for (int size = 0; (sizes[b][size] > 0 && size < n); size++) {
        end = start + sizes[b][size];
        sum = 0;
        val = 1.0 / (scalar_t) sizes[b][size];
        
        for (int i = start; i < end; i++) {
            sum += grad_input[b][i];
        }
        for (int i = start; i < end; i++) {
            ret[b][i] = val * sum;
        }
        start = end;
    }
}

template <typename scalar_t>
__global__ void isotonic_kl_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ret,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sizes,
    int n,
    int batch) {

    int end;
    scalar_t sum;
    scalar_t softmax;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    int start = 0;
    partition(sol, sizes, n, b);
    for (int size = 0; (sizes[b][size] > 0 && size < n); size++) {
        end = start + sizes[b][size];
        sum = 0;
        softmax = 0;
        
        for (int i = start; i < end; i++) {
            softmax += std::exp(s[b][i]);
            sum += grad_input[b][i];
        }
        for (int i = start; i < end; i++) {
            ret[b][i] = std::exp(s[b][i]) / softmax * sum;
        }
        start = end;
    }
}

// === Parallel Implementations =========================================================================

// ---------------------------------------------------------------------------
// Helper types for Phase 3 of isotonic_l2_parallel_kernel.
//
// ValFlag carries a (value, is_block_start) pair through a CUB inclusive
// scan.  PropagateOp propagates values rightward from block starts:
//   if right is a block start → use right's value
//   otherwise               → carry left's value forward
// The operator is associative, so CUB can use its tree-reduction internals.
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct ValFlag {
    scalar_t val;   // representative value of the current PAV block
    int      flag;  // 1 if this position is a PAV block start, else 0
};

template <typename scalar_t>
struct PropagateOp {
    __device__ ValFlag<scalar_t> operator()(
            const ValFlag<scalar_t>& a,
            const ValFlag<scalar_t>& b) const {
        return b.flag ? b : ValFlag<scalar_t>{a.val, 0};
    }
};

// ---------------------------------------------------------------------------
// Optimisations over the original parallel kernel:
//
// 1. target stored as int32 (not scalar_t) — correct semantics, no float-
//    precision issues for large n, no explicit casts in PAV traversal.
//
// 2. sh_last_block[512] in shared memory — after Phase 2a each thread saves
//    the start of the last PAV block in its chunk.  Phase 2b then starts
//    directly at that position instead of traversing from left_start.
//    When sol[last_left] > sol[first_right] (no violation at the boundary),
//    the merge is O(1): just inherit the right region's last block.  This
//    eliminates the dominant O(n) traversal at the final merge level for
//    well-separated inputs.
//
// 3. Parallel Phase 3 via CUB tile-based propagation scan — replaces the
//    serial fill-from-block-start loop with a fully parallel scan using
//    PropagateOp.  TempStorage is O(T) shared memory, independent of n.
//    Works correctly for n >> 1024 via the tiled carry (running_val).
//
// The kernel is always launched with exactly 512 threads (tpb = 512), so
// tpb_p2 is always 256.  For n < 512, extra threads idle in Phase 2a/2b
// and fall through the strided loops in Phases 1 and 3.
// Using 512 instead of 1024 threads doubles the register budget per thread
// (128 vs 64 on SM90), avoiding cudaErrorLaunchOutOfResources with the
// complex VF CUB scan for double precision.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void isotonic_l2_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sums,
    torch::PackedTensorAccessor32<int32_t,  2, torch::RestrictPtrTraits> target,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> c,
    torch::PackedTensorAccessor32<int32_t,  2, torch::RestrictPtrTraits> is_bs,
    int n,
    int batch) {

    // 1KB: last block start per Phase-2a chunk (tpb_p2 = 256 slots).
    __shared__ int sh_last_block[256];

    // O(T) shared memory for the Phase 3 CUB scan, independent of n.
    typedef ValFlag<scalar_t>    VF;
    typedef PropagateOp<scalar_t> PO;
    typedef cub::BlockScan<VF, 512, cub::BLOCK_SCAN_RAKING> Phase3Scan;
    __shared__ typename Phase3Scan::TempStorage phase3_temp;

    const int b      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int tpb    = blockDim.x;   // always 512
    const int tpb_p2 = 256;          // prev power-of-2 of 512

    if (b >= batch) return;

    // ------------------------------------------------------------------
    // Phase 1 (parallel): initialise c, sol, sums, target
    // ------------------------------------------------------------------
    for (int i = tid; i < n; i += tpb) {
        c[b][i]      = static_cast<scalar_t>(1);
        sol[b][i]    = s[b][i];
        sums[b][i]   = s[b][i];
        target[b][i] = i;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2a: each thread runs independent PAV on its chunk,
    //           then records the last block start in sh_last_block.
    // ------------------------------------------------------------------
    if (tid < tpb_p2) {
        const int remainder   = n % tpb_p2;
        const int chunk_base  = n / tpb_p2;
        const int chunk_size  = chunk_base + (tid < remainder ? 1 : 0);
        const int chunk_start = tid * chunk_base + (tid < remainder ? tid : remainder);
        const int chunk_end   = chunk_start + chunk_size;

        int i = chunk_start;
        while (i < chunk_end) {
            int k = target[b][i] + 1;
            if (k >= chunk_end) break;
            if (sol[b][i] > sol[b][k]) { i = k; continue; }
            auto sum_y = sums[b][i];
            auto sum_c = c[b][i];
            while (true) {
                int k_start  = k;                // start of block being absorbed
                auto prev_y  = sol[b][k_start];
                sum_y       += sums[b][k_start];
                sum_c       += c[b][k_start];
                k            = target[b][k_start] + 1;
                is_bs[b][k_start] = 0;           // k_start absorbed; no longer a block start
                if ((k >= chunk_end) || (prev_y > sol[b][k])) {
                    sol[b][i]        = sum_y / sum_c;
                    sums[b][i]       = sum_y;
                    c[b][i]          = sum_c;
                    target[b][i]     = k - 1;
                    target[b][k - 1] = i;
                    if (i > chunk_start) i = target[b][i - 1];
                    break;
                }
            }
        }
        sh_last_block[tid] = i;   // start of last PAV block in this chunk
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2b: binary-tree merge.
    //
    // Key optimisation: start each merge at the LEFT region's last block
    // (sh_last_block[tid]) rather than at left_start.  Since the left
    // region is already non-increasing, the only possible violation is at
    // the boundary with the right region.
    //
    // If sol[last_left] > sol[first_right]: no violation → O(1) update.
    // Otherwise: run PAV from last_left across the combined region.
    // ------------------------------------------------------------------
    for (int stride = 1; stride < tpb_p2; stride <<= 1) {
        if (tid < tpb_p2 && (tid % (2 * stride)) == 0) {
            const int remainder     = n % tpb_p2;
            const int chunk_base    = n / tpb_p2;
            const int right_tid     = tid + stride;
            const int right_tid_end = tid + 2 * stride;

            const int left_start  = tid * chunk_base + (tid < remainder ? tid : remainder);
            const int right_start = right_tid * chunk_base + (right_tid < remainder ? right_tid : remainder);
            const int right_end   = (right_tid_end >= tpb_p2)
                                        ? n
                                        : right_tid_end * chunk_base + (right_tid_end < remainder ? right_tid_end : remainder);

            if (right_start < n) {
                int i = sh_last_block[tid];   // jump to last block of left region
                int k = target[b][i] + 1;    // first block of right region

                if (k >= right_end) {
                    // Right region is empty or left already covers it — nothing to do.
                } else if (sol[b][i] > sol[b][k]) {
                    // No violation at boundary: combined region is already non-increasing.
                    // O(1): inherit right region's last block.
                    sh_last_block[tid] = sh_last_block[right_tid];
                } else {
                    // Violation: run PAV from i across the combined [left_start, right_end).
                    while (i < right_end) {
                        k = target[b][i] + 1;
                        if (k >= right_end) { sh_last_block[tid] = i; break; }
                        if (sol[b][i] > sol[b][k]) { i = k; continue; }
                        auto sum_y = sums[b][i];
                        auto sum_c = c[b][i];
                        while (true) {
                            int k_start  = k;
                            auto prev_y  = sol[b][k_start];
                            sum_y       += sums[b][k_start];
                            sum_c       += c[b][k_start];
                            k            = target[b][k_start] + 1;
                            is_bs[b][k_start] = 0;
                            if ((k >= right_end) || (prev_y > sol[b][k])) {
                                sol[b][i]        = sum_y / sum_c;
                                sums[b][i]       = sum_y;
                                c[b][i]          = sum_c;
                                target[b][i]     = k - 1;
                                target[b][k - 1] = i;
                                if (i > left_start) i = target[b][i - 1];
                                break;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3: parallel reconstruction via tiled CUB propagation scan.
    //
    // After Phase 2b, sol[b][i] is correct only at block starts.  We need
    // to fill every position with its block start's value.
    //
    // We scan (value, is_block_start) pairs with PropagateOp: values flow
    // rightward from block starts, stopping at each new block start.
    //
    // Tiled carry: running_val carries the last propagated value across
    // tile boundaries.  Thread 0 of each tile injects it as a virtual
    // block start when its own position is not a real block start.
    // ------------------------------------------------------------------
    scalar_t running_val = static_cast<scalar_t>(0);  // carry across tiles

    for (int tile_start = 0; tile_start < n; tile_start += tpb) {
        int idx = tile_start + tid;

        VF elem;
        if (idx < n) {
            // is_bs[b][idx] == 1 iff idx is a PAV block start; maintained
            // exactly by the Phase-2a/2b inner loops (absorbed starts cleared to 0).
            bool is_start = (bool)is_bs[b][idx];
            elem.val  = is_start ? sol[b][idx] : static_cast<scalar_t>(0);
            elem.flag = is_start ? 1 : 0;
        } else {
            elem = {static_cast<scalar_t>(0), 0};
        }

        // Thread 0: if not a block start, inject the previous tile's carry
        // so all subsequent non-starts in this tile inherit it correctly.
        if (tid == 0 && !elem.flag) {
            elem.val  = running_val;
            elem.flag = 1;
        }

        VF scan_out, tile_agg;
        Phase3Scan(phase3_temp).InclusiveScan(elem, scan_out, PO(), tile_agg);
        __syncthreads();   // required before reusing phase3_temp next iteration

        if (idx < n) {
            sol[b][idx] = scan_out.val;
        }
        running_val = tile_agg.val;   // broadcast to all threads by CUB
    }
}

// ---------------------------------------------------------------------------
// Backward kernel using cub::BlockScan for the inclusive prefix sum.
//
// Improvements over the naive Hillis-Steele version:
//   - O(N) work scan instead of O(N log N): cub::BlockScan uses a
//     work-efficient algorithm (Blelloch / hybrid) internally.
//   - Shared memory usage is O(T) (thread count), not O(N), so it stays
//     well within the 48 KB limit regardless of sequence length.
//   - pid buffer is int32 instead of scalar_t, avoiding float precision
//     issues for large pool indices.
//   - pid_copy scratch buffer is eliminated entirely.
//
// Always launched with exactly 1024 threads per block (one block per batch
// element).  Threads with tid >= N contribute flag=0 and skip all writes.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void isotonic_l2_backward_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_in,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pid,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> psum,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pcnt,
    int N) {

    // BlockScan TempStorage lives in shared memory: O(T) bytes, independent of N.
    typedef cub::BlockScan<int, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;   // always 1024

    // Phase 1: compute boundary flags and inclusive prefix sum (= pool IDs).
    // Tile-based loop: each tile of T elements is scanned with one BlockScan
    // call, accumulating a running offset across tiles so that pool IDs are
    // globally correct across the full sequence of length N.
    int running_offset = 0;
    for (int tile_start = 0; tile_start < N; tile_start += T) {
        int idx = tile_start + tid;

        // Threads beyond the end of the sequence contribute 0.
        int flag = 0;
        if (idx > 0 && idx < N) {
            flag = (fabsf((float)(sol[b][idx] - sol[b][idx - 1])) > 1e-9f) ? 1 : 0;
        }

        // Inclusive sum within this tile; tile_aggregate is broadcast to all threads.
        int scan_val, tile_aggregate;
        BlockScan(temp_storage).InclusiveSum(flag, scan_val, tile_aggregate);
        __syncthreads();  // required before reusing temp_storage next iteration

        if (idx < N) {
            pid[b][idx] = scan_val + running_offset;
        }
        running_offset += tile_aggregate;
    }
    __syncthreads();

    // Phase 2: zero psum/pcnt accumulators.
    for (int i = tid; i < N; i += T) {
        psum[b][i] = 0.0f;
        pcnt[b][i] = 0.0f;
    }
    __syncthreads();

    // Phase 3: accumulate gradient into per-pool buckets (global atomics).
    for (int i = tid; i < N; i += T) {
        int p = pid[b][i];
        atomicAdd(&psum[b][p], (float)grad_in[b][i]);
        atomicAdd(&pcnt[b][p], 1.0f);
    }
    __syncthreads();

    // Phase 4: write back averaged gradient.
    for (int i = tid; i < N; i += T) {
        int p = pid[b][i];
        out[b][i] = static_cast<scalar_t>(psum[b][p] / (pcnt[b][p] + 1e-12f));
    }
}

// ---------------------------------------------------------------------------
// Numerically stable float atomicMax via CAS loop.
// Works for all finite float values (positive and negative).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int cur = *addr_i;
    while (__int_as_float(cur) < val) {
        int prev = atomicCAS(addr_i, cur, __float_as_int(val));
        if (prev == cur) break;
        cur = prev;
    }
}

// ---------------------------------------------------------------------------
// KL backward parallel kernel — numerically stable via LSE trick.
//
// Structure mirrors isotonic_l2_backward_parallel_kernel; the difference is
// the per-element weight used in Phase 3/4:
//
//   L2:  weight[i] = 1  (uniform average)
//   KL:  weight[i] = softmax(s)[i]  within each pool
//
// Naive sum(exp(s[i])) overflows for large s[i].  Stabilise with LSE:
//
//   lse[p]  = pmax[p] + log( sum_i exp(s[i] - pmax[p]) )
//   out[i]  = exp(s[i] - lse[p]) * pgrad[p]
//
// Two accumulation passes after pool-ID computation:
//   Phase 3a: pmax[p]     = max s[i]              (atomicMax via CAS)
//             pgrad[p]    = sum grad_in[i]         (atomicAdd)
//   Phase 3b: psum_exp[p] = sum exp(s[i]-pmax[p]) (atomicAdd)
//
// Pool boundaries detected identically to L2: positions where sol changes.
// Always launched with 1024 threads per block (one block per batch element).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void isotonic_kl_backward_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_in,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int32_t,  2, torch::RestrictPtrTraits> pid,
    torch::PackedTensorAccessor32<float,    2, torch::RestrictPtrTraits> pgrad,
    torch::PackedTensorAccessor32<float,    2, torch::RestrictPtrTraits> pmax,
    torch::PackedTensorAccessor32<float,    2, torch::RestrictPtrTraits> psum_exp,
    int N) {

    typedef cub::BlockScan<int, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;   // always 1024

    // Phase 1: pool IDs via inclusive prefix sum of boundary flags.
    int running_offset = 0;
    for (int tile_start = 0; tile_start < N; tile_start += T) {
        int idx = tile_start + tid;
        int flag = 0;
        if (idx > 0 && idx < N) {
            flag = (fabsf((float)(sol[b][idx] - sol[b][idx - 1])) > 1e-9f) ? 1 : 0;
        }
        int scan_val, tile_aggregate;
        BlockScan(temp_storage).InclusiveSum(flag, scan_val, tile_aggregate);
        __syncthreads();
        if (idx < N) {
            pid[b][idx] = scan_val + running_offset;
        }
        running_offset += tile_aggregate;
    }
    __syncthreads();

    // Phase 2: initialise per-pool accumulators.
    for (int i = tid; i < N; i += T) {
        pgrad[b][i]    = 0.0f;
        pmax[b][i]     = -1e30f;
        psum_exp[b][i] = 0.0f;
    }
    __syncthreads();

    // Phase 3a: per-pool max of s and gradient sum.
    for (int i = tid; i < N; i += T) {
        int p = pid[b][i];
        atomicMaxFloat(&pmax[b][p], (float)s[b][i]);
        atomicAdd(&pgrad[b][p], (float)grad_in[b][i]);
    }
    __syncthreads();   // pmax must be complete before Phase 3b

    // Phase 3b: shifted exponential sum (numerically stable).
    for (int i = tid; i < N; i += T) {
        int p = pid[b][i];
        atomicAdd(&psum_exp[b][p], expf((float)s[b][i] - pmax[b][p]));
    }
    __syncthreads();

    // Phase 4: softmax-weighted gradient.
    //   lse[p] = pmax[p] + log(psum_exp[p])
    //   out[i] = exp(s[i] - lse[p]) * pgrad[p]
    for (int i = tid; i < N; i += T) {
        int   p     = pid[b][i];
        float lse_p = pmax[b][p] + logf(psum_exp[b][p] + 1e-30f);
        out[b][i]   = static_cast<scalar_t>(expf((float)s[b][i] - lse_p) * pgrad[b][p]);
    }
}

// ---------------------------------------------------------------------------
// Parallel KL isotonic regression forward kernel.
//
// Mirrors isotonic_l2_parallel_kernel exactly; the only changes are:
//   - `sums` / `c` (sum, count) replaced by `lse_y_` / `lse_w_` (log-sum-exp accumulators)
//   - block merge: `sum_y / sum_c`  →  `lse_y - lse_w`
//                  `sum_y += ...`   →  `lse_y = log_add_exp(lse_y, ...)`
//                  `sum_c += ...`   →  `lse_w = log_add_exp(lse_w, ...)`
//   - Phase 1 init: `sol[i] = y[i] - w[i]`, `lse_y_[i] = y[i]`, `lse_w_[i] = w[i]`
//
// All optimisations from the L2 version are retained:
//   1. int32 target (no float-precision issues for large n)
//   2. sh_last_block[256] O(1) boundary optimisation in Phase 2b
//   3. is_bs[] explicit block-start tracking (avoids false-positive heuristic)
//   4. CUB tile-based propagation scan for Phase 3 (O(N) work, O(T) shared mem)
//   5. 512 threads → 128 reg/thread budget (avoids launch-out-of-resources on SM90)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void isotonic_kl_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> lse_y_,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> lse_w_,
    torch::PackedTensorAccessor32<int32_t,  2, torch::RestrictPtrTraits> target,
    torch::PackedTensorAccessor32<int32_t,  2, torch::RestrictPtrTraits> is_bs,
    int n,
    int batch) {

    __shared__ int sh_last_block[256];

    typedef ValFlag<scalar_t>     VF;
    typedef PropagateOp<scalar_t> PO;
    typedef cub::BlockScan<VF, 512, cub::BLOCK_SCAN_RAKING> Phase3Scan;
    __shared__ typename Phase3Scan::TempStorage phase3_temp;

    const int b      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int tpb    = blockDim.x;   // always 512
    const int tpb_p2 = 256;

    if (b >= batch) return;

    // ------------------------------------------------------------------
    // Phase 1: initialise sol, lse_y_, lse_w_, target
    // ------------------------------------------------------------------
    for (int i = tid; i < n; i += tpb) {
        sol[b][i]    = y[b][i] - w[b][i];
        lse_y_[b][i] = y[b][i];
        lse_w_[b][i] = w[b][i];
        target[b][i] = i;
        // is_bs[b][i] already initialised to 1 by the host
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2a: per-chunk PAV; records last block start in sh_last_block.
    // ------------------------------------------------------------------
    if (tid < tpb_p2) {
        const int remainder   = n % tpb_p2;
        const int chunk_base  = n / tpb_p2;
        const int chunk_size  = chunk_base + (tid < remainder ? 1 : 0);
        const int chunk_start = tid * chunk_base + (tid < remainder ? tid : remainder);
        const int chunk_end   = chunk_start + chunk_size;

        int i = chunk_start;
        while (i < chunk_end) {
            int k = target[b][i] + 1;
            if (k >= chunk_end) break;
            if (sol[b][i] > sol[b][k]) { i = k; continue; }
            auto lse_y = lse_y_[b][i];
            auto lse_w = lse_w_[b][i];
            while (true) {
                int k_start = k;
                auto prev_y = sol[b][k_start];
                lse_y       = log_add_exp(lse_y, lse_y_[b][k_start]);
                lse_w       = log_add_exp(lse_w, lse_w_[b][k_start]);
                k           = target[b][k_start] + 1;
                is_bs[b][k_start] = 0;   // k_start absorbed; no longer a block start
                if ((k >= chunk_end) || (prev_y > sol[b][k])) {
                    sol[b][i]        = lse_y - lse_w;
                    lse_y_[b][i]     = lse_y;
                    lse_w_[b][i]     = lse_w;
                    target[b][i]     = k - 1;
                    target[b][k - 1] = i;
                    if (i > chunk_start) i = target[b][i - 1];
                    break;
                }
            }
        }
        sh_last_block[tid] = i;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2b: binary-tree merge (identical structure to L2 parallel).
    // ------------------------------------------------------------------
    for (int stride = 1; stride < tpb_p2; stride <<= 1) {
        if (tid < tpb_p2 && (tid % (2 * stride)) == 0) {
            const int remainder     = n % tpb_p2;
            const int chunk_base    = n / tpb_p2;
            const int right_tid     = tid + stride;
            const int right_tid_end = tid + 2 * stride;

            const int left_start  = tid * chunk_base + (tid < remainder ? tid : remainder);
            const int right_start = right_tid * chunk_base + (right_tid < remainder ? right_tid : remainder);
            const int right_end   = (right_tid_end >= tpb_p2)
                                        ? n
                                        : right_tid_end * chunk_base + (right_tid_end < remainder ? right_tid_end : remainder);

            if (right_start < n) {
                int i = sh_last_block[tid];
                int k = target[b][i] + 1;

                if (k >= right_end) {
                    // Right region empty — nothing to do.
                } else if (sol[b][i] > sol[b][k]) {
                    // No violation at boundary: O(1) inherit right's last block.
                    sh_last_block[tid] = sh_last_block[right_tid];
                } else {
                    // Violation: run PAV from i across the combined region.
                    while (i < right_end) {
                        k = target[b][i] + 1;
                        if (k >= right_end) { sh_last_block[tid] = i; break; }
                        if (sol[b][i] > sol[b][k]) { i = k; continue; }
                        auto lse_y = lse_y_[b][i];
                        auto lse_w = lse_w_[b][i];
                        while (true) {
                            int k_start = k;
                            auto prev_y = sol[b][k_start];
                            lse_y       = log_add_exp(lse_y, lse_y_[b][k_start]);
                            lse_w       = log_add_exp(lse_w, lse_w_[b][k_start]);
                            k           = target[b][k_start] + 1;
                            is_bs[b][k_start] = 0;
                            if ((k >= right_end) || (prev_y > sol[b][k])) {
                                sol[b][i]        = lse_y - lse_w;
                                lse_y_[b][i]     = lse_y;
                                lse_w_[b][i]     = lse_w;
                                target[b][i]     = k - 1;
                                target[b][k - 1] = i;
                                if (i > left_start) i = target[b][i - 1];
                                break;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3: parallel reconstruction via CUB propagation scan.
    // Identical to isotonic_l2_parallel_kernel Phase 3.
    // ------------------------------------------------------------------
    scalar_t running_val = static_cast<scalar_t>(0);

    for (int tile_start = 0; tile_start < n; tile_start += tpb) {
        int idx = tile_start + tid;

        VF elem;
        if (idx < n) {
            bool is_start = (bool)is_bs[b][idx];
            elem.val  = is_start ? sol[b][idx] : static_cast<scalar_t>(0);
            elem.flag = is_start ? 1 : 0;
        } else {
            elem = {static_cast<scalar_t>(0), 0};
        }

        if (tid == 0 && !elem.flag) {
            elem.val  = running_val;
            elem.flag = 1;
        }

        VF scan_out, tile_agg;
        Phase3Scan(phase3_temp).InclusiveScan(elem, scan_out, PO(), tile_agg);
        __syncthreads();

        if (idx < n) {
            sol[b][idx] = scan_out.val;
        }
        running_val = tile_agg.val;
    }
}

// === Host Functions ===================================================================================

// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2(torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));
    auto batch = y.size(0);
    auto n = y.size(1);
    auto sol = torch::zeros_like(y);
    auto sums = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    auto c = torch::zeros_like(y);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_l2", ([&] {
        isotonic_l2_kernel<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sums.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return sol;
}

// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2_parallel(torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));

    auto batch = y.size(0);
    auto n     = y.size(1);

    auto sol    = torch::zeros_like(y);
    auto sums   = torch::zeros_like(y);
    // target stored as int32: correct semantics, no float-precision issues
    // for large n, and matches the int32_t kernel parameter.
    auto target = torch::zeros({batch, n}, torch::dtype(torch::kInt32).device(y.device()));
    auto c      = torch::zeros_like(y);
    // is_bs[b][i] == 1 iff position i is a PAV block start after Phase 2.
    // Initialized to all-ones (every position is its own singleton block start).
    // PAV inner loops clear absorbed starts to 0.
    auto is_bs  = torch::ones({batch, n}, torch::dtype(torch::kInt32).device(y.device()));

    // Always 512 threads to match cub::BlockScan<VF, 512> template parameter.
    // 512 threads gives 128 registers/thread budget (vs 64 at 1024), needed
    // to avoid cudaErrorLaunchOutOfResources with the complex VF scan kernel.
    const int tpb    = 512;
    const int blocks = batch;   // one block = one sequence

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_l2_parallel", ([&] {
        isotonic_l2_parallel_kernel<scalar_t><<<blocks, tpb>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sums.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            is_bs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));

    return sol;
}

// Solves isotonic optimization with KL divergence using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
torch::Tensor isotonic_kl(torch::Tensor y, torch::Tensor w) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));
    auto batch = y.size(0);
    auto n = y.size(1);
    auto sol = torch::zeros_like(y);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    auto target = torch::zeros_like(y);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_kl", ([&] {
        isotonic_kl_kernel<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lse_y_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lse_w_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return sol;
}

torch::Tensor isotonic_l2_backward(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(s));
    auto batch = sol.size(0);
    auto n = sol.size(1);
    auto ret = torch::zeros_like(sol);
    auto sizes = torch::zeros_like(sol);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_l2_backward", ([&] {
        isotonic_l2_backward_kernel<scalar_t><<<blocks, threads>>>(
            s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sizes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return ret;
}

torch::Tensor isotonic_l2_backward_parallel(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(sol));
    const int B = sol.size(0);
    const int N = sol.size(1);

    // Always 1024 threads to match the cub::BlockScan<int, 1024> template.
    const int tpb    = 1024;
    const int blocks = B;

    auto ret     = torch::zeros_like(sol);
    // pid uses int32 — avoids float precision issues for large pool indices
    // and eliminates the old pid_copy scratch buffer entirely.
    auto pid_buf = torch::empty({B, N}, torch::dtype(torch::kInt32).device(sol.device()));

    auto float_opts = torch::dtype(torch::kFloat32).device(sol.device());
    auto psum_buf   = torch::zeros({B, N}, float_opts);
    auto pcnt_buf   = torch::zeros({B, N}, float_opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_l2_backward_parallel", ([&] {
        isotonic_l2_backward_parallel_kernel<scalar_t><<<blocks, tpb>>>(
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            pid_buf.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            psum_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pcnt_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            N
        );
    }));

    return ret;
}

torch::Tensor isotonic_kl_backward(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(s));
    auto batch = sol.size(0);
    auto n = sol.size(1);
    auto ret = torch::zeros_like(sol);
    auto sizes = torch::zeros_like(sol);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_kl_backward", ([&] {
        isotonic_kl_backward_kernel<scalar_t><<<blocks, threads>>>(
            s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sizes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return ret;
}


torch::Tensor isotonic_kl_backward_parallel(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(sol));
    const int B = sol.size(0);
    const int N = sol.size(1);

    const int tpb    = 1024;
    const int blocks = B;

    auto ret          = torch::zeros_like(sol);
    auto pid_buf      = torch::empty({B, N}, torch::dtype(torch::kInt32).device(sol.device()));
    auto float_opts   = torch::dtype(torch::kFloat32).device(sol.device());
    auto pgrad_buf    = torch::zeros({B, N}, float_opts);
    auto pmax_buf     = torch::empty({B, N}, float_opts);  // initialised to -1e30 inside kernel
    auto psum_exp_buf = torch::zeros({B, N}, float_opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_kl_backward_parallel", ([&] {
        isotonic_kl_backward_parallel_kernel<scalar_t><<<blocks, tpb>>>(
            s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            pid_buf.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            pgrad_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pmax_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            psum_exp_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            N
        );
    }));

    return ret;
}

// Solves isotonic optimization with KL divergence using parallel PAV.
// Same interface as isotonic_kl but dispatches isotonic_kl_parallel_kernel.
torch::Tensor isotonic_kl_parallel(torch::Tensor y, torch::Tensor w) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));

    auto batch = y.size(0);
    auto n     = y.size(1);

    auto sol    = torch::zeros_like(y);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    // int32 target: correct semantics, no float-precision issues for large n
    auto target = torch::zeros({batch, n}, torch::dtype(torch::kInt32).device(y.device()));
    // is_bs: 1 for every position initially (all singletons are block starts)
    auto is_bs  = torch::ones({batch, n}, torch::dtype(torch::kInt32).device(y.device()));

    // 512 threads: matches cub::BlockScan<VF, 512> template parameter and
    // gives 128 reg/thread budget on SM90 (avoids launch-out-of-resources).
    const int tpb    = 512;
    const int blocks = batch;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_kl_parallel", ([&] {
        isotonic_kl_parallel_kernel<scalar_t><<<blocks, tpb>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lse_y_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lse_w_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            is_bs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));

    return sol;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isotonic_l2", &isotonic_l2, "Isotonic L2");
  m.def("isotonic_l2_backward", &isotonic_l2_backward, "Isotonic L2 Backward");
  m.def("isotonic_l2_parallel", &isotonic_l2_parallel, "Isotonic L2 Parallel");
  m.def("isotonic_l2_backward_parallel", &isotonic_l2_backward_parallel, "Isotonic L2 Backward Parallel");
  m.def("isotonic_kl", &isotonic_kl, "Isotonic KL");
  m.def("isotonic_kl_backward", &isotonic_kl_backward, "Isotonic KL Backward");
  m.def("isotonic_kl_parallel", &isotonic_kl_parallel, "Isotonic KL Parallel");
  m.def("isotonic_kl_backward_parallel", &isotonic_kl_backward_parallel, "Isotonic KL Backward Parallel");
}
