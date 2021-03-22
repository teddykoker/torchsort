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
#include <algorithm>
#include <cmath>

//  Copied from fast-soft-sort (https://bit.ly/3r0gOav) with the following modifications:
//  - replace numpy functions with torch equivalents
//  - re-write in C++
//  - return solution in place
//  - added backward pass (vector jacobian product)

//  Copied from scikit-learn with the following modifications:
//  - use decreasing constraints by default,
//  - do not return solution in place, rather save in array `sol`,
//  - avoid some needless multiplications.

// Numerically stable log-add-exp
template <typename scalar_t>
inline scalar_t log_add_exp(scalar_t x, scalar_t y) {
    scalar_t larger = std::max(x, y);
    scalar_t smaller = std::min(x, y);
    return larger + std::log1p(std::exp(smaller - larger));
}

// Returns partition corresponding to solution."""
template <typename scalar_t>
std::vector<int> partition(torch::TensorAccessor<scalar_t, 1> solution, int n) {
    const scalar_t eps = 1.0e-9;

    if (n == 0) {
        return std::vector<int>();
    }

    std::vector<int> sizes{1};
    
    for (int i = 1; i < n; i++) {
        if (std::abs(solution[i] - solution[i - 1]) > eps) {
            sizes.push_back(0);
        }
        sizes[sizes.size() - 1] += 1;
    }
    return sizes;
}


template <typename scalar_t>
void isotonic_l2_kernel(
    torch::TensorAccessor<scalar_t, 2> s,
    torch::TensorAccessor<scalar_t, 2> sol,
    torch::TensorAccessor<scalar_t, 2> sums,
    torch::TensorAccessor<scalar_t, 2> target,
    torch::TensorAccessor<scalar_t, 2> c,
    int n,
    int batch) {

    // #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
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
}

template <typename scalar_t>
void isotonic_kl_kernel(
    torch::TensorAccessor<scalar_t, 2> y,
    torch::TensorAccessor<scalar_t, 2> w,
    torch::TensorAccessor<scalar_t, 2> sol,
    torch::TensorAccessor<scalar_t, 2> lse_y_,
    torch::TensorAccessor<scalar_t, 2> lse_w_,
    torch::TensorAccessor<scalar_t, 2> target,
    int n,
    int batch) {

    // #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
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
}


template <typename scalar_t>
void isotonic_l2_backward_kernel(
    torch::TensorAccessor<scalar_t, 2> s, // not used
    torch::TensorAccessor<scalar_t, 2> sol,
    torch::TensorAccessor<scalar_t, 2> grad_input,
    torch::TensorAccessor<scalar_t, 2> ret,
    int n,
    int batch) {

    int end;
    scalar_t sum;
    scalar_t val;

    // #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
        int start = 0;
        for (int size: partition(sol[b], n)) {
            end = start + size;
            sum = 0;
            val = 1.0 / (scalar_t) size;
            
            for (int i = start; i < end; i++) {
                sum += grad_input[b][i];
            }
            for (int i = start; i < end; i++) {
                ret[b][i] = val * sum;
            }
            start = end;
        }
    }
}

template <typename scalar_t>
void isotonic_kl_backward_kernel(
    torch::TensorAccessor<scalar_t, 2> s,
    torch::TensorAccessor<scalar_t, 2> sol,
    torch::TensorAccessor<scalar_t, 2> grad_input,
    torch::TensorAccessor<scalar_t, 2> ret,
    int n,
    int batch) {

    int end;
    scalar_t sum;
    scalar_t softmax;

    // #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
        int start = 0;
        for (int size: partition(sol[b], n)) {
            end = start + size;
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
}

// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2(torch::Tensor y) {
    auto batch = y.size(0);
    auto n = y.size(1);
    auto sol = torch::zeros_like(y);
    auto sums = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    auto c = torch::zeros_like(y);

    AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_l2", ([&] {
        isotonic_l2_kernel<scalar_t>(
            y.accessor<scalar_t, 2>(),
            sol.accessor<scalar_t, 2>(),
            sums.accessor<scalar_t, 2>(),
            target.accessor<scalar_t, 2>(),
            c.accessor<scalar_t, 2>(),
            n,
            batch);
    }));
    return sol;
}

// Solves isotonic optimization with KL divergence using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
torch::Tensor isotonic_kl(torch::Tensor y, torch::Tensor w) {
    auto batch = y.size(0);
    auto n = y.size(1);
    auto sol = torch::zeros_like(y);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    auto target = torch::zeros_like(y);

    AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_kl", ([&] {
        isotonic_kl_kernel<scalar_t>(
            y.accessor<scalar_t, 2>(),
            w.accessor<scalar_t, 2>(),
            sol.accessor<scalar_t, 2>(),
            lse_y_.accessor<scalar_t, 2>(),
            lse_w_.accessor<scalar_t, 2>(),
            target.accessor<scalar_t, 2>(),
            n,
            batch);
    }));
    return sol;
}

torch::Tensor isotonic_l2_backward(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    auto batch = sol.size(0);
    auto n = sol.size(1);
    auto ret = torch::zeros_like(sol);

    AT_DISPATCH_FLOATING_TYPES(sol.scalar_type(), "isotonic_l2_backward", ([&] {
        isotonic_l2_backward_kernel<scalar_t>(
            s.accessor<scalar_t, 2>(),
            sol.accessor<scalar_t, 2>(),
            grad_input.accessor<scalar_t, 2>(),
            ret.accessor<scalar_t, 2>(),
            n,
            batch);
    }));
    return ret;
}

torch::Tensor isotonic_kl_backward(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    auto batch = sol.size(0);
    auto n = sol.size(1);
    auto ret = torch::zeros_like(sol);

    AT_DISPATCH_FLOATING_TYPES(sol.scalar_type(), "isotonic_kl_backward", ([&] {
        isotonic_kl_backward_kernel<scalar_t>(
            s.accessor<scalar_t, 2>(),
            sol.accessor<scalar_t, 2>(),
            grad_input.accessor<scalar_t, 2>(),
            ret.accessor<scalar_t, 2>(),
            n,
            batch);
    }));
    return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isotonic_l2", &isotonic_l2, "Isotonic L2");
  m.def("isotonic_l2_backward", &isotonic_l2_backward, "Isotonic L2 Backward");
  m.def("isotonic_kl", &isotonic_kl, "Isotonic KL");
  m.def("isotonic_kl_backward", &isotonic_kl_backward, "Isotonic KL Backward");
}
