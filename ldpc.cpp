/*
 * This file is part of the simulator_awgn_python distribution
 * https://github.com/and-kirill/ldpc_soft_py/.
 * Copyright (c) 2023 Kirill Andreev.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cmath>
#include <vector>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "./ldpc.h"

#define ABS(A) (((A) >= 0) ? (A) : -(A))
#define HARD(A) (((A) < 0) ? 1 : 0)
#define OFFSET(A, B) (((A) > (B)) ? ((A)-(B)) : 0)
#define INFTY 1000000


// Matrix implementation
template<typename T>
Matrix<T>::Matrix(index_t nrows, index_t ncols) :
  m_nrows(nrows),
  m_ncols(ncols),
  m_buf(new T[m_nrows * m_ncols])
{}

template<typename T>
Matrix<T>::Matrix(index_t nrows, index_t ncols, T fill_val) :
  m_nrows(nrows),
  m_ncols(ncols),
  m_buf(new T[m_nrows * m_ncols])
{
  fill_with(fill_val);
}

template<typename T>
Matrix<T>::~Matrix() {
  delete[] m_buf;
}

template<typename T>
void Matrix<T>::fill_with(T fill_val) {
  for (index_t i = 0; i < m_ncols * m_nrows; i++ ) {
    m_buf[i] = fill_val;
  }
}

template<typename T>
const T& Matrix<T>::operator()(std::pair<index_t, index_t>index) const {
  return operator()(index.first, index.second);
}

template<typename T>
T& Matrix<T>::operator()(std::pair<index_t, index_t>index) {
  return operator()(index.first, index.second);
}

template<typename T>
T& Matrix<T>::operator()(index_t row_ind, index_t col_ind) {
  return m_buf[row_ind * m_ncols + col_ind];
}

template<typename T>
const T& Matrix<T>::operator()(index_t row_ind, index_t col_ind) const {
  return m_buf[row_ind * m_ncols + col_ind];
}

// Tanner graph implementation
TannerGraph::TannerGraph(index_t blklen,
                         index_t nchks,
                         index_t max_cw,
                         index_t max_rw) :
  n(blklen),
  m(nchks),
  cmax(max_cw),
  rmax(max_rw),
  col_weight(std::vector<index_t>(n)),
  row_weight(std::vector<index_t>(m)),
  row_col(Matrix<index_t>(m, rmax, 0)),
  col_row(Matrix<index_t>(n, cmax, 0)),
  col_N(Matrix<index_t>(n, cmax, 0)),
  msgs_col(Matrix<std::pair<index_t, index_t> >(n, cmax, std::make_pair(0, 0)))
{
  std::fill(col_weight.begin(), col_weight.end(), 0);
  std::fill(row_weight.begin(), row_weight.end(), 0);
}

// Alist loader implementation
std::vector<index_t>read_line(std::ifstream& fp,
                              index_t        expected_len,
                              std::string    action) {
  std::string line;

  if (!std::getline(fp, line)) {
    std::cout << "File corrupted. Failed at " << action << ". Exiting." <<
      std::endl;
    return std::vector<index_t>();
  }
  std::vector<index_t> line_vec;
  std::istringstream   ss(line);
  index_t num;

  while (ss >> num) {
    line_vec.push_back(num);
  }

  if (line_vec.size() != expected_len) {
    std::cout << "Line length mismatch at " << action << "." << std::endl;
    return std::vector<index_t>();
  }
  return line_vec;
}

TannerGraph* load_alist(const char *filename) {
  std::ifstream fp(filename);

  if (fp.fail()) {
    std::cout << "ERROR: Cannot open file " << filename << "." << std::endl;
    return 0;
  }
  std::vector<index_t> data = read_line(fp, 2, "reading matrix size");

  if (!data.size()) return 0;
  index_t n = data[0];
  index_t m = data[1];

  data = read_line(fp, 2, "reading row/column max weights");

  if (!data.size()) return 0;
  index_t cmax     = data[0];
  index_t rmax     = data[1];
  TannerGraph *tng = new TannerGraph(n, m, cmax, rmax);

  // Reading row and column weights
  data = read_line(fp, n, "reading column weights");

  if (!data.size()) return 0;
  tng->col_weight.assign(data.begin(), data.end());

  data = read_line(fp, m, "reading row weights");

  if (!data.size()) return 0;
  tng->row_weight.assign(data.begin(), data.end());

  // Per-column representation is skipped
  for (index_t i = 0; i < n; i++ ) {
    data = read_line(fp, cmax, "reading matrix column");

    if (!data.size()) return 0;
  }

  // Get data from per-row matrix representation
  std::vector<index_t> count = std::vector<index_t>(n);
  std::fill(count.begin(), count.end(), 0);

  for (index_t i = 0; i < m; i++) {
    data = read_line(fp, rmax, "reading matrix row");

    if (!data.size()) return 0;

    for (index_t j = 0; j < tng->row_weight[i]; j++) {
      index_t v = data[j] - 1;
      tng->row_col(i, j)        = v;
      tng->col_row(v, count[v]) = i;
      tng->col_N(v, count[v])   = j;
      count[v]++;
    }
  }

  for (index_t i = 0; i < n; i++) {
    for (index_t j = 0; j < tng->col_weight[i]; j++) {
      tng->msgs_col(i, j) = std::make_pair(tng->col_row(i, j),
                                           tng->col_N(i, j));
    }
  }
  return tng;
}

TannerGraph* load_pcm(int* pcm, int n_rows, int n_cols) {
    int cmax = 0;
    for (int i = 0; i < n_cols; i++) {
        int current_col_weight = 0;
        for (int j = 0; j < n_rows; j++) {
            current_col_weight += pcm[j * n_cols + i];
        }
        if (current_col_weight > cmax) {
            cmax = current_col_weight;
        }
    }

    int rmax = 0;
    for (int i = 0; i < n_rows; i++) {
        int current_row_weight = 0;
        for (int j = 0; j < n_cols; j++) {
            current_row_weight += pcm[i * n_cols + j];
        }
        if (current_row_weight > rmax) {
            rmax = current_row_weight;
        }
    }

    TannerGraph *tng = new TannerGraph(n_cols, n_rows, cmax, rmax);

    for (int i = 0; i < n_cols; i++) {
        int current_col_weight = 0;
        for (int j = 0; j < n_rows; j++) {
            current_col_weight += pcm[j * n_cols + i];
        }
        tng->col_weight[i] = current_col_weight;
    }

    for (int i = 0; i < n_rows; i++) {
        int current_row_weight = 0;
        for (int j = 0; j < n_cols; j++) {
            current_row_weight += pcm[i * n_cols + j];
        }
        tng->row_weight[i] = current_row_weight;
    }

  std::vector<index_t> count = std::vector<index_t>(n_cols);
  std::fill(count.begin(), count.end(), 0);

  for (int i = 0; i < n_rows; i++) {
    int current_row_weight = 0;
    for (int j = 0; j < n_cols; j++) {
        if (pcm[i * n_cols + j] == 1) {
            tng->row_col(i, current_row_weight) = j;
            tng->col_row(j, count[j]) = i;
            tng->col_N(j, count[j]) = current_row_weight;
            count[j]++;
            current_row_weight++;
        }
    }
  }

  for (index_t i = 0; i < n_cols; i++) {
    for (index_t j = 0; j < tng->col_weight[i]; j++) {
      tng->msgs_col(i, j) = std::make_pair(tng->col_row(i, j),
                                           tng->col_N(i, j));
    }
  }
  return tng;
}

void layered_min_sum(const TannerGraph         & tng,
                     const std::vector<double> & llr_in,
                     index_t                     n_iter,
                     const std::vector<index_t>& row_seq,
                     const std::vector<double> & scales,
                     const std::vector<double> & offsets,
                     double                     *llr_out,
                     double                     *llr_out_intermediate)
{
  Matrix<double> r_msg(tng.m, tng.rmax, 0);

  for (index_t i = 0; i < tng.n; i++) {
    llr_out[i] = llr_in[i];
  }

  for (index_t loop = 0; loop < n_iter; ++loop)
  {
    for (index_t t = 0; t < tng.m; t++)
    {
      index_t j          = row_seq[t];
      double  first_min  = INFTY;
      double  second_min = INFTY;
      index_t min_index  = 0;
      int     sum_sign   = 0;

      for (index_t k = 0; k < tng.row_weight[j]; k++)
      {
        double q_msg = llr_out[tng.row_col(j, k)] - r_msg(j, k);
        double temp  = fabs(q_msg);

        if (temp < first_min)
        {
          second_min = first_min;
          first_min  = temp;
          min_index  = k;
        }
        else if (temp < second_min)
        {
          second_min = temp;
        }

        if (q_msg < 0)
        {
          sum_sign ^= 1;
        }
      }

      for (index_t k = 0; k < tng.row_weight[j]; k++)
      {
        int sign     = sum_sign;
        double q_msg = llr_out[tng.row_col(j, k)] - r_msg(j, k);

        if (q_msg < 0)
        {
          sign ^= 1;
        }
        llr_out[tng.row_col(j, k)] = q_msg;

        if (k == min_index)
        {
          r_msg(j, k) = second_min;
        }
        else
        {
          r_msg(j, k) = first_min;
        }
        double scale  = scales[tng.row_col(j, k)];
        double offset = offsets[tng.row_col(j, k)];
        r_msg(j, k) = (1 - 2 * sign) * scale *
                      OFFSET(r_msg(j, k), offset);
        llr_out[tng.row_col(j, k)] += r_msg(j, k);
      }
    } // Loop over parity checks
    for (index_t i = 0; i < tng.n; i++) {
      llr_out_intermediate[loop*tng.n+i] = llr_out[i];
    }
    
  }   // Loop over iteration
}

// Min-Sum Decoder
void min_sum(const TannerGraph        & ldpc,
             const std::vector<double>& llr_in,
             index_t                    n_iter,
             const std::vector<double>& scales,
             const std::vector<double>& offsets,
             double                    *llr_out,
             double                    *llr_out_intermediate)
{
  // Auxiliary matrices
  Matrix<double> r_msg(ldpc.m, ldpc.rmax); // messages from check to variable
                                           // nodes
  Matrix<double> q_msg(ldpc.m, ldpc.rmax); // messages from variable to check

  // nodes

  // Initialization
  for (index_t i = 0; i < ldpc.n; ++i)
  {
    llr_out[i] = llr_in[i];

    for (index_t j = 0; j < ldpc.col_weight[i]; ++j)
    {
      q_msg(ldpc.msgs_col(i, j)) = llr_in[i];
    }
  }

  for (index_t loop = 0; loop < n_iter; ++loop)
  {
    // Update R messages
    for (index_t j = 0; j < ldpc.m; j++)
    {
      double  first_min  = INFTY;
      double  second_min = INFTY;
      index_t min_index  = 0;
      int     sum_sign   = 0;

      for (index_t k = 0; k < ldpc.row_weight[j]; k++)
      {
        double temp = fabs(q_msg(j, k));

        if (temp < first_min)
        {
          second_min = first_min;
          first_min  = temp;
          min_index  = k;
        }
        else if (temp < second_min)
        {
          second_min = temp;
        }

        if (q_msg(j, k) < 0)
        {
          sum_sign ^= 1;
        }
      }

      for (index_t k = 0; k < ldpc.row_weight[j]; k++)
      {
        int sign = sum_sign;

        if (q_msg(j, k) < 0)
        {
          sign ^= 1;
        }

        if (k == min_index)
        {
          r_msg(j, k) = second_min;
        }
        else
        {
          r_msg(j, k) = first_min;
        }
        double scale  = scales[ldpc.row_col(j, k)];
        double offset = offsets[ldpc.row_col(j, k)];
        r_msg(j, k) = (1 - 2 * sign) * scale * OFFSET(r_msg(j, k), offset);
      }
    }

    // Update Q messages and output LLRs
    for (index_t i = 0; i < ldpc.n; i++)
    {
      llr_out[i] = llr_in[i];

      for (index_t k = 0; k < ldpc.col_weight[i]; k++)
      {
        llr_out[i] += r_msg(ldpc.msgs_col(i, k));
      }

      for (index_t k = 0; k < ldpc.col_weight[i]; k++)
      {
        q_msg(ldpc.msgs_col(i, k)) = llr_out[i] - r_msg(ldpc.msgs_col(i, k));
      }
      llr_out_intermediate[loop*ldpc.n+i] = llr_out[i];
    } // Loop over variable nodes
  }   // Loop over decoding iterations
}

static double logtanh(double x) {
  static const double MAX_ARG = 31.;
  static const double MIN_ARG = -log(tanh(MAX_ARG / 2));

  // Perform clipping
  if (x > MAX_ARG) {
    return 0;
  } else if (x < MIN_ARG) {
    return MAX_ARG; // logtanh(logtanh(x)) == x
  }

  // Use approximation at large values
  if (x > 12.5) {
    return 2 * exp(-x);
  }
  return -log(tanh(x / 2));
}

void sum_product(const TannerGraph        & ldpc,
                 const std::vector<double>& llr_in,
                 index_t                    n_iter,
                 double                    *llr_out,
                 double                    *llr_out_intermediate)
{
  // Messages from check to variable nodes
  Matrix<double> r_msg(ldpc.m, ldpc.rmax);

  // Messages from variable to check nodes: signs and log-magnitudes are stored
  // separately. Note that log-magnitudes are required to avoid products of
  // hyperbolic tangents
  Matrix<double> q_ltanh(ldpc.m, ldpc.rmax);
  Matrix<int>    q_signs(ldpc.m, ldpc.rmax);
  double sum_ltanh = 0;
  int    sum_sign  = 0;
  int    sign      = 0;
  double temp      = 0;

  // Initialization
  for (index_t i = 0; i < ldpc.n; ++i)
  {
    llr_out[i] = llr_in[i];

    for (index_t j = 0; j < ldpc.col_weight[i]; ++j)
    {
      q_signs(ldpc.msgs_col(i, j)) = 0;

      if (llr_in[i] < 0)
      {
        q_signs(ldpc.msgs_col(i, j)) = 1;
      }
      q_ltanh(ldpc.msgs_col(i, j)) = logtanh(fabs(llr_in[i]));
    }
  }

  // Decoding iterations
  for (index_t loop = 0; loop < n_iter; ++loop)
  {
    // Update R messages
    for (index_t j = 0; j < ldpc.m; j++)
    {
      sum_ltanh = 0;
      sum_sign  = 0;

      for (index_t k = 0; k < ldpc.row_weight[j]; k++)
      {
        sum_ltanh += q_ltanh(j, k);
        sum_sign  ^= q_signs(j, k);
      }

      for (index_t k = 0; k < ldpc.row_weight[j]; k++)
      {
        sign        = sum_sign ^ q_signs(j, k);
        r_msg(j, k) = (1 - 2 * sign) * logtanh(sum_ltanh - q_ltanh(j, k));
      }
    } // Loop over check nodes

    // Update Q messages
    for (index_t i = 0; i < ldpc.n; i++)
    {
      llr_out[i] = llr_in[i];

      for (index_t k = 0; k < ldpc.col_weight[i]; k++)
      {
        llr_out[i] += r_msg(ldpc.msgs_col(i, k));
      }

      for (index_t k = 0; k < ldpc.col_weight[i]; k++)
      {
        temp = llr_out[i] - r_msg(ldpc.msgs_col(i, k));
        q_signs(ldpc.msgs_col(i, k)) = 0;

        if (temp < 0)
        {
          q_signs(ldpc.msgs_col(i, k)) = 1;
        }
        q_ltanh(ldpc.msgs_col(i, k)) = logtanh(fabs(temp));
      }
      llr_out_intermediate[loop*ldpc.n+i] = llr_out[i];
    } // Loop over variable nodes
  }   // Loop over decoding iterations
}
