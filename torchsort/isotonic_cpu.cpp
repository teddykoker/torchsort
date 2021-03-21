#include <torch/extension.h>
#include <algorithm>
#include <cmath>

template <typename scalar_t>
inline scalar_t log_add_exp(scalar_t x, scalar_t y) {
    scalar_t larger = std::max(x, y);
    scalar_t smaller = std::min(x, y);
    return larger + std::log1p(std::exp(smaller - larger));
}

template <typename scalar_t>
void isotonic_l2_kernel(
    torch::TensorAccessor<scalar_t, 1> y,
    torch::TensorAccessor<scalar_t, 1> sol,
    torch::TensorAccessor<scalar_t, 1> sums,
    torch::TensorAccessor<scalar_t, 1> target,
    torch::TensorAccessor<scalar_t, 1> c,
    int n) {
    // target describes a list of blocks.  at any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        c[i] = 1.0;
        sol[i] = y[i];
        sums[i] = y[i];
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

// Solves isotonic optimization with KL divergence using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
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

torch::Tensor isotonic_l2(torch::Tensor y) {
    auto n = y.size(0);
    auto sol = torch::zeros_like(y);
    auto sums = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    auto c = torch::zeros_like(y);
    switch (y.type().scalarType()) {
        case torch::ScalarType::Double:
            isotonic_l2_kernel<double>(
                y.accessor<double, 1>(),
                sol.accessor<double, 1>(),
                sums.accessor<double, 1>(),
                target.accessor<double, 1>(),
                c.accessor<double, 1>(),
                n);
            break;
        case torch::ScalarType::Float:
            isotonic_l2_kernel<float>(
                y.accessor<float, 1>(),
                sol.accessor<float, 1>(),
                sums.accessor<float, 1>(),
                target.accessor<float, 1>(),
                c.accessor<float, 1>(),
                n);
            break;
    }
    return sol;
}

torch::Tensor isotonic_kl(torch::Tensor y, torch::Tensor w) {
    auto n = y.size(0);
    auto sol = torch::zeros_like(y);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    switch (y.type().scalarType()) {
        case torch::ScalarType::Double:
            isotonic_kl_kernel<double>(
                y.accessor<double, 1>(),
                w.accessor<double, 1>(),
                sol.accessor<double, 1>(),
                lse_y_.accessor<double, 1>(),
                lse_w_.accessor<double, 1>(),
                target.accessor<double, 1>(),
                n);
            break;
        case torch::ScalarType::Float:
            isotonic_kl_kernel<float>(
                y.accessor<float, 1>(),
                w.accessor<float, 1>(),
                sol.accessor<float, 1>(),
                lse_y_.accessor<float, 1>(),
                lse_w_.accessor<float, 1>(),
                target.accessor<float, 1>(),
                n);
            break;
    }
    return sol;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isotonic_l2", &isotonic_l2, "Isotonic L2");
  m.def("isotonic_kl", &isotonic_kl, "Isotonic KL");
}
