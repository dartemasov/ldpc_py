/*
 * This file is part of the simulator_awgn_python distribution
 * https://github.com/and-kirill/ldpc_soft_py/.
 * Copyright (c) 2023 Kirill Andreev, Alexey Frolov
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


#ifndef LDPC_H_
#define LDPC_H_

#include <vector>

/// Define the type for all indices
typedef int index_t;

/**
 * Matrix class. Supports a MATLAB-style access by index x(i, j);
 */
template<typename T>
class Matrix {
public:

  /**
   * Initialize matrix with the number of rows, columns
   * Specify a default value to fill all elements with
   * @param nrows The number of rows in the matrix
   * @param ncols The number of columns in the matrix
   */
  Matrix(index_t nrows,
         index_t ncols);

  /**
   * Constructor with default value
   */
  Matrix(index_t nrows,
         index_t ncols,
         T       fill_val);

  virtual ~Matrix();

  /// Fill all elements of the matrix with given value
  void     fill_with(T value);

  /// Index access operators access: x(row_index, col_index)
  const T& operator()(std::pair<index_t, index_t>indx) const;
  T&       operator()(std::pair<index_t, index_t>indx);
  T&       operator()(index_t row_ind,
                      index_t col_ind);
  const T& operator()(index_t row_ind,
                      index_t col_ind) const;

private:

  /// The number of rows
  index_t m_nrows;

  /// The number of columns
  index_t m_ncols;

  /// Raw buffer of length nrows X ncols
  T *m_buf;

  std::vector<T *>m_row_ptrs;
};


struct TannerGraph {
  /**
   * Initialize an empty Tanner graph representation
   * The constructor generates empty matrices (filled with zeros),
   * which are filled by load_alist() function.
   * See doc.tex for more explanation
   * @param blklen block length
   * @param nchks  The number of parity checks
   * @param max_cw Maximum number of nonzero column entries
   * @param max_rw Maximum number of nonzero row entries
   */
  TannerGraph(index_t blklen,
              index_t nchks,
              index_t max_cw,
              index_t max_rw);

  /// Block length (the number of parity check matrix columns)
  index_t n;

  /// The number of parity checks
  index_t m;

  /// Maximum number of nonzero entries in each column
  index_t cmax;

  /// Maximum number of nonzero row entries
  index_t rmax;

  /// Column weights, a vector of length n
  std::vector<index_t>col_weight;

  /// Row weights, a vector of length m
  std::vector<index_t>row_weight;

  /// positions of ones in rows
  Matrix<index_t>row_col;

  /// positions of ones in columns
  Matrix<index_t>col_row;

  /// Positions of ones in row_col
  Matrix<index_t>col_N;

  /// This pair shows the position of msg in R_msg and Q_msg
  Matrix<std::pair<index_t, index_t> >msgs_col;
};


/// Load Tanner graph from the alist file
TannerGraph* load_alist(const char *filename);

/**
 * Layered Min-sum decoder. Layered processing assumes that
 * output LLRs are updated after each parity check processing.
 * The layered decoding algorithm is the following:
 *  1. Iterate over decoding iterations
 *  2. Iterate over all parity checks
 *  3. Process single parity check
 * There are also scaling and offset coefficients represented as vectors
   Each weight set is a vector of block length
 * The output message magnitudes from check to variable nodes
 * are scaled and offset
 * @param tng     Pointer to the Tanner graph structure
 * @param llr_in vector of channel log likelihood ratios
 * @param n_iter  The number of decoding iterations
 * @param row_seq is a parity checks processing schedule:
 *        array of check node indices
 * @param scales multiplicative scales
 * @param offsets applies relu(x - offset) for each message magnitude
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */

void layered_min_sum(const TannerGraph         & tng,
                     const std::vector<double> & llr_in,
                     index_t                     n_iter,
                     const std::vector<index_t>& row_seq,
                     const std::vector<double> & scales,
                     const std::vector<double> & offsets,
                     double                     *llr_out,
                     double                     *llr_out_intermediate);

/**
 * Min-sum decoder. Layered processing assumes that
 * output LLRs are updated after each parity check processing.
 * Non-layered processing assumes:
 * 1. Initialization
 * 2. For each decoding iteration:
 *  - update R-messages: from check to variable nodes
 *  - Update Q-messages: from variable to check nodes
 *  - Update output LLRs.
 * @param tng     Pointer to the Tanner graph structure
 * @param llr_in vector of channel log likelihood ratios
 * @param n_iter  The number of decoding iterations
 * @param scales multiplicative scales
 * @param offsets applies relu(x - offset) for each message magnitude
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */
void min_sum(const TannerGraph        & tng,
             const std::vector<double>& llr_in,
             index_t                    n_iter,
             const std::vector<double>& scales,
             const std::vector<double>& offsets,
             double                    *llr_out,
             double                    *llr_out_intermediate);

/**
 * Sum-product decoder. To avoid a product of hyperbolic tangents,
 * the calculations are performed over the logarithm of the LLR magnitudes.
 * @param tng     Pointer to the Tanner graph structure
 * @param llr_in vector of channel log likelihood ratios
 * @param n_iter  The number of decoding iterations
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */

void sum_product(const TannerGraph        & ldpc,
                 const std::vector<double>& llr_in,
                 index_t                    n_iter,
                 double                    *llr_out,
                 double                    *llr_out_intermediate);

#endif  // LDPC_H_
