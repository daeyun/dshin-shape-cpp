///////////////////////////////////////////////////////////////////////////////
//
//  Name:        mexutil.h
//  Purpose:     Macros and helper functions for creating MATLAB MEX-files.
//  Author:      Daeyun Shin <dshin11@illinois.edu>
//  Created:     01.15.2015
//  Modified:    03.18.2015
//  Version:     0.1.1
//
//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "mex.h"
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef N_LHS_VAR
#define N_LHS_VAR nargout
#endif
#ifndef N_RHS_VAR
#define N_RHS_VAR nargin
#endif
#ifndef MEX_COMPONENT_NAME
#define MEX_COMPONENT_NAME "MATLAB"
#endif

#define BOLD(str) "<strong>" str "</strong>"
#define ORANGE(str) "[\b" str "]\b"

namespace mexutil {
enum CompOp { EQ, GT, LT, NEQ, GE, LE };

enum ArgType {
  kDouble,
  kSingle,
  kStruct,
  kLogical,
  kChar,
  kInt8,
  kUint8,
  kInt16,
  kUint16,
  kInt32,
  kUint32
};

struct StructField {
  std::string field_name;
  ArgType type;
  size_t dims[3];
  void *value;
};

struct MatlabStruct {
  std::vector<StructField> fields;
};

void createStructArray(const std::vector<MatlabStruct> &structs, mxArray *&out,
                       bool is_copy = true);

// Redirect stderr to a file or stringstream .
void CaptureErrorMsg(std::stringstream &stderr_content);
void CaptureErrorMsg(const std::string &filename);

// e.g. double* mat = GetArg<kDouble,EQ,GT>(0, prhs, 3, 3); Throws an error if
// prhs[0] doesn't have exactly 3 rows or have less than 3 columns. 0s are
// ignored.
template <ArgType argtype, CompOp row_comp = EQ, CompOp col_comp = EQ>
void *GetArg(const mwSize index, const mxArray *input[], mwSize nrows = 0,
             mwSize ncols = 0);

// Constructs the identifier token used in error messages.
std::string MatlabIdStringFromFilename(std::string str);
std::string FilenameFromPath(std::string str);

// Retrieves the workspace global variable mexVerboseLevel (default: 1).
int VerboseLevel();

const int kDefaultVerboseLevel = 1;
const std::string kFilename = FilenameFromPath(__FILE__);
const std::string kFunctionIdentifier = MatlabIdStringFromFilename(kFilename);
const int kVerboseLevel = VerboseLevel();

// Force pass-by-value behavior to prevent accidentally modifying shared
// memory content in-place. Undocumented.
// http://undocumentedmatlab.com/blog/matlab-mex-in-place-editing
extern "C" bool mxUnshareArray(mxArray *array_ptr, bool noDeepCopy);

// Copy and transpose.
template <size_t nrows_in, typename T>
void Transpose(const std::vector<T> &in, T *out);

// Useful when zero-based indexing is used.
template <size_t nrows_in, typename T>
void TransposeAddOne(const std::vector<T> &in, T *out);

mxArray *UnshareArray(int index, const mxArray *prhs[]) {
  mxArray *unshared = const_cast<mxArray *>(prhs[index]);
  mxUnshareArray(unshared, true);
  return unshared;
}

std::string MatlabIdStringFromFilename(std::string str) {
  (void)(MatlabIdStringFromFilename);
  auto is_invalid_id_char =
      [](char ch) { return !(isalnum((int)ch) || ch == '_'); };
  if (int i = str.find_first_of('.')) str = str.substr(0, i);
  if (!isalpha(str[0])) str = "mex_" + str;
  std::replace_if(str.begin(), str.end(), is_invalid_id_char, '_');
  return str;
}

std::string FilenameFromPath(std::string str) {
  (void)(FilenameFromPath);
  if (int i = str.find_last_of('/')) str = str.substr(i + 1, str.length());
  return str;
}

int VerboseLevel() {
  (void)(VerboseLevel);
  mxArray *ptr = mexGetVariable("global", "mexVerboseLevel");
  if (ptr == NULL) return kDefaultVerboseLevel;
  return mxGetScalar(ptr);
}

void CaptureErrorMsg(std::stringstream &stderr_content) {
  std::cerr.rdbuf(stderr_content.rdbuf());
}

void CaptureErrorMsg(const std::string &filename) {
  freopen(filename.c_str(), "a", stderr);
}

template <size_t nrows_in, typename T>
void Transpose(const std::vector<T> &in, T *out) {
  const size_t ncols_in = in.size() / nrows_in;
#pragma omp parallel for
  for (size_t i = 0; i < in.size(); i += nrows_in) {
    for (size_t j = 0; j < nrows_in; ++j) {
      *(out + (i / nrows_in) + ncols_in * j) = in[i + j];
    }
  }
}

template <size_t nrows_in, typename T>
void TransposeAddOne(const std::vector<T> &in, T *out) {
  const size_t ncols_in = in.size() / nrows_in;
#pragma omp parallel for
  for (size_t i = 0; i < in.size(); i += nrows_in) {
    for (size_t j = 0; j < nrows_in; ++j) {
      *(out + (i / nrows_in) + ncols_in * j) = in[i + j] + 1;
    }
  }
}

// e.g. LEVEL(2, MPRINTF("Not printed if logging level is less than 2."))
#define LEVEL(verbose_level, expr)            \
  {                                           \
    if (kVerboseLevel >= verbose_level) expr; \
  }

// Construct an identifier string e.g.  MATLAB:mexutil:myErrorIdentifier
#define MEX_IDENTIFIER(mnemonic)                               \
  (std::string(MEX_COMPONENT_NAME ":") + kFunctionIdentifier + \
   std::string(":" mnemonic)).c_str()

// Assert number of input variables.
#define N_IN_RANGE(min, max)                                                \
  {                                                                         \
    if (N_RHS_VAR < min || N_RHS_VAR > max) {                               \
      mexErrMsgIdAndTxt(MEX_IDENTIFIER("InputSizeError"),                   \
                        "Number of inputs must be between %d and %d.", min, \
                        max);                                               \
    }                                                                       \
  }

// Assert number of output variables.
#define N_OUT_RANGE(min, max)                                                \
  {                                                                          \
    if (N_LHS_VAR < min || N_LHS_VAR > max) {                                \
      mexErrMsgIdAndTxt(MEX_IDENTIFIER("OutputSizeError"),                   \
                        "Number of outputs must be between %d and %d.", min, \
                        max);                                                \
    }                                                                        \
  }

#define N_IN(num)                                             \
  {                                                           \
    if (N_RHS_VAR != num) {                                   \
      mexErrMsgIdAndTxt(MEX_IDENTIFIER("InputSizeError"),     \
                        "Number of inputs must be %d.", num); \
    }                                                         \
  }

#define N_OUT(num)                                             \
  {                                                            \
    if (N_LHS_VAR != num) {                                    \
      mexErrMsgIdAndTxt(MEX_IDENTIFIER("OutputSizeError"),     \
                        "Number of outputs must be %d.", num); \
    }                                                          \
  }

#define VAR(name)                                                          \
  {                                                                        \
    std::ostringstream val_str;                                            \
    val_str << name;                                                       \
    mexutil::DisplayVariable(#name, val_str.str(), sizeof(name),           \
                             (void *)&name, __FILE__, __LINE__, __func__); \
  }

#define CHECK(name)                                                           \
  {                                                                           \
    std::ostringstream val_str;                                               \
    val_str << name;                                                          \
    mexutil::DisplayVariable(#name, val_str.str(), sizeof(name), 0, __FILE__, \
                             __LINE__, __func__);                             \
  }

// Print message to MATLAB console.
// e.g.MPRINTF(BOLD("%d"), argc);
#define MPRINTF(...)           \
  {                            \
    mexPrintf(__VA_ARGS__);    \
    mexEvalString("drawnow;"); \
  }

// Display error and exit.
#define ERR_EXIT(errname, ...) \
  { mexErrMsgIdAndTxt(MEX_IDENTIFIER(errname), ##__VA_ARGS__); }

// Macros starting with an underscore are internal.
#define _M_ASSERT(condition)                                                 \
  {                                                                          \
    if (!(condition)) {                                                      \
      MPRINTF("[ERROR] (%s:%d %s) ", kFilename.c_str(), __LINE__, __func__); \
      mexErrMsgTxt("assertion " #condition " failed\n");                     \
    }                                                                        \
  }

#define _M_ASSERT_MSG(condition, msg)                                        \
  {                                                                          \
    if (!(condition)) {                                                      \
      MPRINTF("[ERROR] (%s:%d %s) ", kFilename.c_str(), __LINE__, __func__); \
      mexErrMsgIdAndTxt(MEX_IDENTIFIER("AssertionError"),                    \
                        "assertion " #condition " failed\n%s\n", msg);       \
    }                                                                        \
  }

#define _CHOOSE_MACRO(a, x, func, ...) func

#define M_ASSERT(condition, ...)                                        \
  _CHOOSE_MACRO(, ##__VA_ARGS__, _M_ASSERT_MSG(condition, __VA_ARGS__), \
                _M_ASSERT(condition))

#define M_ASSERT_FMT(condition, fmt, ...)                                    \
  {                                                                          \
    if (!(condition)) {                                                      \
      MPRINTF("[ERROR] (%s:%d %s) ", kFilename.c_str(), __LINE__, __func__); \
      mexErrMsgIdAndTxt(MEX_IDENTIFIER("AssertionError"),                    \
                        "assertion " #condition " failed\n" fmt "\n",        \
                        ##__VA_ARGS__);                                      \
    }                                                                        \
  }

template <ArgType argtype, CompOp row_comp, CompOp col_comp>
void *GetArg(mwSize index, const mxArray *input[], mwSize nrows, mwSize ncols) {
  if (nrows > 0) {
    switch (row_comp) {
      case EQ:
        M_ASSERT_FMT(mxGetM(input[index]) == nrows,
                     "size(input[%d], 1) must be %d.", index, nrows);
        break;
      case GT:
        M_ASSERT_FMT(mxGetM(input[index]) > nrows,
                     "size(input[%d], 1) must be greater than %d.", index,
                     nrows);
        break;
      case LT:
        M_ASSERT_FMT(mxGetM(input[index]) < nrows,
                     "size(input[%d], 1) must be less than %d.", index, nrows);
        break;
      case NEQ:
        M_ASSERT_FMT(mxGetM(input[index]) != nrows,
                     "size(input[%d], 1) must be not equal %d.", index, nrows);
        break;
      case GE:
        M_ASSERT_FMT(mxGetM(input[index]) >= nrows,
                     "size(input[%d], 1) must be at least %d.", index, nrows);
        break;
      case LE:
        M_ASSERT_FMT(mxGetM(input[index]) <= nrows,
                     "size(input[%d], 1) can be at most %d.", index, nrows);
        break;
      default:
        break;
    }
  }
  if (ncols > 0) {
    switch (col_comp) {
      case EQ:
        M_ASSERT_FMT(mxGetN(input[index]) == ncols,
                     "size(input[%d], 2) must be %d.", index, ncols);
        break;
      case GT:
        M_ASSERT_FMT(mxGetN(input[index]) > ncols,
                     "size(input[%d], 2) must be greater than %d.", index,
                     ncols);
        break;
      case LT:
        M_ASSERT_FMT(mxGetN(input[index]) < ncols,
                     "size(input[%d], 2) must be less than %d.", index, ncols);
        break;
      case NEQ:
        M_ASSERT_FMT(mxGetN(input[index]) != ncols,
                     "size(input[%d], 2) must be not equal %d.", index, ncols);
        break;
      case GE:
        M_ASSERT_FMT(mxGetN(input[index]) >= ncols,
                     "size(input[%d], 2) must be at least %d.", index, ncols);
        break;
      case LE:
        M_ASSERT_FMT(mxGetN(input[index]) <= ncols,
                     "size(input[%d], 2) can be at most %d.", index, ncols);
        break;
      default:
        break;
    }
  }

  switch (argtype) {
    case kDouble:
      M_ASSERT_FMT(mxIsDouble(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetPr(input[index]);  // double*
    case kSingle:
      M_ASSERT_FMT(mxIsSingle(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // float*
    case kStruct:
      M_ASSERT_FMT(mxIsStruct(input[index]),
                   "Invalid data type for input index %d.", index);
      // TODO
      ERR_EXIT("UnknownDataTypeError", "Not implemented");
      break;
    case kLogical:
      M_ASSERT_FMT(mxIsLogical(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetLogicals(input[index]);  // mxLogical*
    case kChar:
      M_ASSERT_FMT(mxIsChar(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetChars(input[index]);  // char*
    case kInt8:
      M_ASSERT_FMT(mxIsInt8(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // int8_t*
    case kUint8:
      M_ASSERT_FMT(mxIsUint8(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // uint8_t*
    case kInt16:
      M_ASSERT_FMT(mxIsInt16(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // int16_t*
    case kUint16:
      M_ASSERT_FMT(mxIsUint16(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // uint16_t*
    case kInt32:
      M_ASSERT_FMT(mxIsInt32(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // int32_t*
    case kUint32:
      M_ASSERT_FMT(mxIsUint32(input[index]),
                   "Invalid data type for input index %d.", index);
      return mxGetData(input[index]);  // uint32_t*
    default:
      ERR_EXIT("UnknownDataTypeError", "Unknown argtype");
  }
}

void DisplayVariable(std::string name, std::string value, size_t size,
                     void *ptr, std::string file, int line, std::string func) {
  if (ptr == 0) {
    MPRINTF("[INFO]  (%s:%d %s)  %s=%s  %d\n", file.c_str(), line, func.c_str(),
            name.c_str(), value.c_str(), (int)size);
  } else {
    MPRINTF("[INFO]  (%s:%d %s)  %s=%s  &%s=%p  %d\n", file.c_str(), line,
            func.c_str(), name.c_str(), value.c_str(), name.c_str(), ptr,
            (int)size);
  }
}

/**
 * @brief Create a MATLAB struct array from a vector of MatlabStructs.
 *
 * @param[in] structs
 * @param[out] out
 * @param[in] is_copy default: true
 *
 * Example usage:
 * @code
 *   std::vector<double> values = {3.14, 42, 9, 7};
 *   createStructArray({
 *        {{
 *          {"a", kDouble, {1, 3, 1}, &values[0]},
 *          {"b", kDouble, {1, 2, 1}, &values[1]},
 *        }},
 *        {{
 *          {"a", kDouble, {1, 3, 1}, &values[0]},
 *          {"b", kDouble, {2, 2, 1}, &values[0]},
 *        }},
 *       }, out[0]);
 * @endcode
 */
void createStructArray(const std::vector<MatlabStruct> &structs, mxArray *&out,
                       bool is_copy) {
  mwSize num_structs = structs.size();
  mwSize dims[2] = {1, num_structs};

  size_t num_fields = structs[0].fields.size();
  std::vector<const char *> field_names;
  std::transform(structs[0].fields.begin(), structs[0].fields.end(),
                 std::back_inserter(field_names), [](const StructField &field) {
    return field.field_name.c_str();
  });

  mxArray *struct_array =
      mxCreateStructArray(2, dims, num_fields, &field_names[0]);

  for (size_t i = 0; i < num_structs; i++) {
    const MatlabStruct &s = structs[i];
    M_ASSERT(num_fields == s.fields.size());
    for (size_t j = 0; j < num_fields; j++) {
      M_ASSERT(s.fields[j].field_name == structs[0].fields[j].field_name);
      M_ASSERT(s.fields[j].type == structs[0].fields[j].type);

      mxArray *value;
      switch (s.fields[j].type) {
        case kDouble:
          value = mxCreateDoubleMatrix(s.fields[j].dims[0], s.fields[j].dims[1],
                                       mxREAL);
          if (is_copy) {
            std::copy((double *)s.fields[j].value,
                      (double *)s.fields[j].value +
                          s.fields[j].dims[0] * s.fields[j].dims[1],
                      (double *)mxGetPr(value));
          } else {
            mxSetPr(value, (double *)s.fields[j].value);
          }
          break;
        case kInt32:
          value = mxCreateNumericMatrix(
              s.fields[j].dims[0], s.fields[j].dims[1], mxINT32_CLASS, mxREAL);
          if (is_copy) {
            std::copy((int32_t *)s.fields[j].value,
                      (int32_t *)s.fields[j].value +
                          s.fields[j].dims[0] * s.fields[j].dims[1],
                      (int32_t *)mxGetPr(value));
          } else {
            mxSetData(value, s.fields[j].value);
          }
          break;
        case kLogical:
          value =
              mxCreateLogicalMatrix(s.fields[j].dims[0], s.fields[j].dims[1]);
          if (is_copy) {
            mxLogical * data_ptr = mxGetLogicals(value);
            for (size_t k = 0; k<s.fields[j].dims[0] * s.fields[j].dims[1];k++) {
              data_ptr[k] = *((int *) s.fields[j].value + k);
            }
          } else {
            mxSetData(value, s.fields[j].value);
          }
          break;
        case kStruct:
          createStructArray(*((std::vector<MatlabStruct> *)s.fields[j].value),
                            value, is_copy);
          break;
        default:
          ERR_EXIT("UnsupportedFieldTypeError", "Unsupported field type");
          return;
      }

      mxSetField(struct_array, i, field_names[j], value);
    }
  }

  out = struct_array;
}

}  // end of namespace mexutil
