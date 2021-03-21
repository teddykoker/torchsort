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

template <typename scalar_t>
void isotonic_l2_kernel(
    torch::TensorAccessor<scalar_t, 1> s,
    torch::TensorAccessor<scalar_t, 1> sol,
    torch::TensorAccessor<scalar_t, 1> sums,
    torch::TensorAccessor<scalar_t, 1> target,
    torch::TensorAccessor<scalar_t, 1> c,
    int n) {
    // target describes a list of blocks.  at any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        c[i] = 1.0;
        sol[i] = s[i];
        sums[i] = s[i];
        target[i] = i;
    }

    int i = 0;
    while (i < n) {
        auto k = target[i] + 1;
        if (k == n) {
            break;
        }
        if (sol[i] > sol[k]) {
            i = k;
            continue;
        }
        auto sum_y = sums[i];
        auto sum_c = c[i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[k];
            sum_y += sums[k];
            sum_c += c[k];
            k = target[k] + 1;
            if ((k == n) || (prev_y > sol[k])) {
                // Non-singleton increasing subsequence is finished,
                // update first entry.
                sol[i] = sum_y / sum_c;
                sums[i] = sum_y;
                c[i] = sum_c;
                target[i] = k - 1;
                target[k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can.  This makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[i - 1];
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[i] + 1;
        for (int j = i + 1; j < k; j++) {
            sol[j] = sol[i];
        }
        i = k;
    }
}

template <typename scalar_t>
void isotonic_kl_kernel(
    torch::TensorAccessor<scalar_t, 1> y,
    torch::TensorAccessor<scalar_t, 1> w,
    torch::TensorAccessor<scalar_t, 1> sol,
    torch::TensorAccessor<scalar_t, 1> lse_y_,
    torch::TensorAccessor<scalar_t, 1> lse_w_,
    torch::TensorAccessor<scalar_t, 1> target,
    int n) {
    // target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        sol[i] = y[i] - w[i];
        lse_y_[i] = y[i];
        lse_w_[i] = w[i];
        target[i] = i;
    }

    int i = 0;
    while (i < n) {
        auto k = target[i] + 1;
        if (k == n) {
            break;
        }
        if (sol[i] > sol[k]) {
            i = k;
            continue;
        }
        auto lse_y = lse_y_[i];
        auto lse_w = lse_w_[i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[k];
            lse_y = log_add_exp(lse_y, lse_y_[k]);
            lse_w = log_add_exp(lse_w, lse_w_[k]);
            k = target[k] + 1;
            if ((k == n) || (prev_y > sol[k])) {
                // Non-singleton increasing subsequence is finished,
                // update first entry.
                sol[i] = lse_y - lse_w;
                lse_y_[i] = lse_y;
                lse_w_[i] = lse_w;
                target[i] = k - 1;
                target[k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can.  This makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[i - 1];
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[i] + 1;
        for (int j = i + 1; j < k; j++) {
            sol[j] = sol[i];
        }
        i = k;
    }
}

// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2(torch::Tensor y) {
    auto n = y.size(0);
    auto sol = torch::zeros_like(y);
    auto sums = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    auto c = torch::zeros_like(y);

    AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_l2", ([&] {
        isotonic_l2_kernel<scalar_t>(
            y.accessor<scalar_t, 1>(),
            sol.accessor<scalar_t, 1>(),
            sums.accessor<scalar_t, 1>(),
            target.accessor<scalar_t, 1>(),
            c.accessor<scalar_t, 1>(),
            n);
    }));
    return sol;
}

// Solves isotonic optimization with KL divergence using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
torch::Tensor isotonic_kl(torch::Tensor y, torch::Tensor w) {
    auto n = y.size(0);
    auto sol = torch::zeros_like(y);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    auto target = torch::zeros_like(y);

    AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_kl", ([&] {
        isotonic_kl_kernel<scalar_t>(
            y.accessor<scalar_t, 1>(),
            w.accessor<scalar_t, 1>(),
            sol.accessor<scalar_t, 1>(),
            lse_y_.accessor<scalar_t, 1>(),
            lse_w_.accessor<scalar_t, 1>(),
            target.accessor<scalar_t, 1>(),
            n);
    }));
    return sol;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isotonic_l2", &isotonic_l2, "Isotonic L2");
  m.def("isotonic_kl", &isotonic_kl, "Isotonic KL");
}
