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

#include <iostream>
#include <cmath>
#include "./ldpc.h"


extern "C"
void* init_ldpc(char *alist_path) {
  return load_alist(alist_path);
}

extern "C"
void free_ldpc(void *ldpc_ptr) {
  delete (static_cast<TannerGraph *>(ldpc_ptr));
}

extern "C"
void decode_soft(index_t  command,
                 void    *tng_ptr,
                 double  *llr_in,
                 index_t  n_iterations,
                 double  *llr_out_buf,
                 index_t *row_seq,
                 double  *scale_array,
                 double  *offset_array,
                 double  *llr_out_intermediate) {
  TannerGraph *tng = static_cast<TannerGraph *>(tng_ptr);

  switch (command) {
    case 0:
      sum_product(*tng,
                  std::vector<double>(llr_in, llr_in + tng->n),
                  n_iterations,
                  llr_out_buf,
                  llr_out_intermediate);
      return;
    case 2:
      min_sum(*tng,
              std::vector<double>(llr_in,       llr_in + tng->n),
              n_iterations,
              std::vector<double>(scale_array,  scale_array + tng->n),
              std::vector<double>(offset_array, offset_array + tng->n),
              llr_out_buf,
              llr_out_intermediate);
      return;
    case 3:
      layered_min_sum(*tng,
                      std::vector<double>( llr_in,       llr_in + tng->n),
                      n_iterations,
                      std::vector<index_t>(row_seq,      row_seq + tng->m),
                      std::vector<double>( scale_array,  scale_array + tng->n),
                      std::vector<double>( offset_array, offset_array + tng->n),
                      llr_out_buf,
                      llr_out_intermediate);
      return;
    default:
      std::cout << "Command " << command << " not supported." << std::endl;
      return;
  }
}
