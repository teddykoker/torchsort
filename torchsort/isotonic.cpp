
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
#include <iostream>

using namespace torch::indexing;
using namespace std;

//  Copied from fast-soft-sort (https://bit.ly/3r0gOav) with the following modifications:
//  - replace numpy functions with torch equivalents
//  - re-write in C++
//  - return solution in place

//  Copied from scikit-learn with the following modifications:
//  - use decreasing constraints by default,
//  - do not return solution in place, rather save in array `sol`,
//  - avoid some needless multiplications.


// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2(torch::Tensor y) {
    auto n = y.size(0);
    auto target = torch::arange(n);
    auto c = torch::ones_like(y);
    auto sums = torch::ones_like(y);
    auto sol = torch::zeros_like(y);

    // target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    
    for (int i = 0; i < n; i++) {
        sol[i] = y[i];
        sums[i] = y[i];
    }

    int i = 0;
    while (i < n) {
        auto k = target[i].item<int>() + 1;
        if (k == n) {
            break;
        }
        if (sol[i].item<double>() > sol[k].item<double>()) {
            i = k;
            continue;
        }
        auto sum_y = sums[i];
        auto sum_c = c[i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[k].item<double>();
            sum_y += sums[k];
            sum_c += c[k];
            k = target[k].item<int>() + 1;
            if ((k == n) || (prev_y > sol[k].item<double>())) {
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
                    i = target[i - 1].item<int>();
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[i].item<int64_t>() + 1;
        sol.index_put_({Slice(i + 1, k, None)}, sol[i]);
        i = k;
    }
    return sol;
}
    

// Numerically stable log-add-exp
torch::Tensor log_add_exp(torch::Tensor x, torch::Tensor y) {
    auto larger = torch::max(x, y);
    auto smaller = torch::min(x, y);
    return larger + torch::log1p(torch::exp(smaller - larger));
}


// Solves isotonic optimization with KL divergence using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
torch::Tensor isotonic_kl(torch::Tensor y, torch::Tensor w) {
    auto n = y.size(0);
    auto target = torch::arange(n);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    auto sol = torch::zeros_like(y);

    // target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.

    for (int i = 0; i < n; i++) {
        sol[i] = y[i] - w[i];
        lse_y_[i] = y[i];
        lse_w_[i] = w[i];
    }

    int i = 0;
    while (i < n) {
        auto k = target[i].item<int>() + 1;
        if (k == n) {
            break;
        }
        if (sol[i].item<double>() > sol[k].item<double>()) {
            i = k;
            continue;
        }
        auto lse_y = lse_y_[i];
        auto lse_w = lse_w_[i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[k].item<double>();
            lse_y = log_add_exp(lse_y, lse_y_[k]);
            lse_w = log_add_exp(lse_w, lse_w_[k]);
            k = target[k].item<int>() + 1;
            if ((k == n) || (prev_y > sol[k].item<double>())) {
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
                    i = target[i - 1].item<int>();
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[i].item<int64_t>() + 1;
        sol.index_put_({Slice(i + 1, k, None)}, sol[i]);
        i = k;
    }
    return sol;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isotonic_l2", &isotonic_l2, "Isotonic L2");
  m.def("isotonic_kl", &isotonic_kl, "Isotonic KL");
}
