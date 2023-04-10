#include <iomanip>
#include <iostream>

#include "ark/file_io.h"
#include "ark/ops/ops_test_utils.h"
#include "ark/random.h"

using namespace std;

// Return an array of range values.
template <typename T>
unique_ptr<T[]> range_array(size_t num, float begin, float diff)
{
    T *ret = new T[num];
    for (size_t i = 0; i < num; ++i) {
        ret[i] = T(begin);
        begin += diff;
    }
    return unique_ptr<T[]>(ret);
}

// Calculate the error rate between two values.
template <typename T> float error_rate(T a, T b)
{
    T diff = abs(a - b);
    if (diff < numeric_limits<T>::min()) {
        return 0;
    }
    diff -= numeric_limits<T>::epsilon();
    T half_eps = numeric_limits<T>::epsilon() * T(0.5);
    if (a > b) {
        a -= half_eps;
        b += half_eps;
    } else {
        a += half_eps;
        b -= half_eps;
    }
    return (float)diff / max(abs((float)a), abs((float)b));
}

// Return mean squared error and max error rate between two matrices.
template <typename T>
pair<float, float> cmp_matrix(T *ground_truth, T *res, unsigned int m,
                              unsigned int n, unsigned int bs, unsigned int lm,
                              unsigned int ln, bool print)
{
    if (lm == 0) {
        lm = m;
    }
    if (ln == 0) {
        ln = n;
    }
    size_t num = (size_t)lm * (size_t)ln;

    const float thres_err = 0.01;

    float l2_loss = 0;
    float max_err = 0;
    // int cnt_flip = 0;
    // float max_err_gv;
    // float max_err_rv;
    for (unsigned int bidx = 0; bidx < bs; ++bidx) {
        for (unsigned int nidx = 0; nidx < n; ++nidx) {
            for (unsigned int midx = 0; midx < m; ++midx) {
                unsigned int idx = midx + nidx * lm + bidx * lm * ln;
                T gv = ground_truth[idx];
                T rv = res[idx];
                float diff = (float)(gv - rv);
                l2_loss += diff * diff;
                float err = error_rate(gv, rv);
                // if ((err > thres_err) && (error_rate(gv, -rv) < thres_err) &&
                // (((float)gv * (float)rv) < 0)) {
                //     cnt_flip++;
                //     cout << (float)gv << "," << (float)rv << endl;
                //     cout << hex << gv.storage << "," << rv.storage << dec <<
                //     endl;
                // }
                if (err > max_err) {
                    max_err = err;
                    // max_err_gv = (float)gv;
                    // max_err_rv = (float)rv;
                }
            }
        }
    }
    if (print) {
        unsigned int x = 0;
        unsigned int cc = 0;
        cout << setprecision(4);
        for (unsigned int bidx = 0; bidx < bs; ++bidx) {
            for (unsigned int nidx = 0; nidx < n; ++nidx) {
                for (unsigned int midx = 0; midx < m; ++midx) {
                    unsigned int idx = midx + nidx * lm + bidx * lm * ln;
                    T exp = ground_truth[idx];
                    T act = res[idx];
                    if (error_rate(exp, act) < thres_err) {
                        cout << (float)act << ',';
                    } else {
                        cout << "\033[0;31m" << (float)act << "\033[0m,"
                             << "\033[0;32m" << (float)exp << "\033[0m,";
                    }
                    if (++cc == m) {
                        cout << '[' << x << ']' << endl;
                        cc = 0;
                        x++;
                    }
                }
            }
        }
    }
    // cout << max_err_gv << endl;
    // cout << max_err_rv << endl;
    // cout << cnt_flip << endl;
    return {l2_loss / num, max_err};
}

//
template <typename T>
void print_matrix(T *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln)
{
    unsigned int x = 0;
    unsigned int cc = 0;
    cout << setprecision(4);
    for (unsigned int bidx = 0; bidx < bs; ++bidx) {
        for (unsigned int nidx = 0; nidx < n; ++nidx) {
            for (unsigned int midx = 0; midx < m; ++midx) {
                unsigned int idx = midx + nidx * lm + bidx * lm * ln;
                T v = val[idx];
                cout << (float)v << ',';
                if (++cc == m) {
                    cout << '[' << x << ']' << endl;
                    cc = 0;
                    x++;
                }
            }
        }
    }
}

// Return a random half_t array.
unique_ptr<half_t[]> rand_halfs(size_t num, float max_val)
{
    return rand_array<half_t>(num, max_val);
}

// Return a random float array.
unique_ptr<float[]> rand_floats(size_t num, float max_val)
{
    return rand_array<float>(num, max_val);
}

// Return a half_t range array.
unique_ptr<half_t[]> range_halfs(size_t num, float begin, float diff)
{
    return range_array<half_t>(num, begin, diff);
}

// Return a float range array.
unique_ptr<float[]> range_floats(size_t num, float begin, float diff)
{
    return range_array<float>(num, begin, diff);
}

//
float error_rate(half_t a, half_t b)
{
    return error_rate<half_t>(a, b);
}

//
float error_rate(float a, float b)
{
    return error_rate<float>(a, b);
}

//
pair<float, float> cmp_matrix(half_t *ground_truth, half_t *res, unsigned int m,
                              unsigned int n, unsigned int bs, unsigned int lm,
                              unsigned int ln, bool print)
{
    return cmp_matrix<half_t>(ground_truth, res, m, n, bs, lm, ln, print);
}

//
pair<float, float> cmp_matrix(float *ground_truth, float *res, unsigned int m,
                              unsigned int n, unsigned int bs, unsigned int lm,
                              unsigned int ln, bool print)
{
    return cmp_matrix<float>(ground_truth, res, m, n, bs, lm, ln, print);
}

//
string get_kernel_code(const string &name)
{
    return ark::read_file(ark::get_dir(string{__FILE__}) + "/kernels/" + name +
                          ".h");
}
