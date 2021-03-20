#include <torch/extension.h>
#include <iostream>

using namespace torch::indexing;
using namespace std;


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
            // Non-singleton increasing subsequence is finished,
            // update first entry.
            auto prev_y = sol[k].item<double>();
            sum_y += sums[k];
            sum_c += c[k];
            k = target[k].item<int>() + 1;
            if ((k == n) || (prev_y > sol[k].item<double>())) {
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
    
//*
// Numerically stable log-add-exp
//
torch::Tensor log_add_exp(torch::Tensor x, torch::Tensor y) {
    auto larger = torch::max(x, y);
    auto smaller = torch::min(x, y);
    return larger + torch::log1p(torch::exp(smaller - larger));
}

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
