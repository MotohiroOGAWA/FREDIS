/*
 * msentity-similarity C API — used by the Python, Java, and C# bindings.
 *
 * Conventions are those of msentity's C API (msentity/c/msentity_c.h):
 * int32 status codes (MSENTITY_OK = 0, MSENTITY_ERR_*), a per-thread error
 * message (mssim_last_error), UTF-8 strings freed with mssim_string_free,
 * and msentity_value cells released with mssim_values_clear.
 *
 * Datasets are passed as the bytes of an .msds file, produced by
 * msentity_to_msds_bytes() of msentity's C API (or any .msds file read into
 * memory). Results (mssim_result*) correspond to .mssim files.
 */
#ifndef MSENTITY_SIMILARITY_C_H
#define MSENTITY_SIMILARITY_C_H

#include <stddef.h>
#include <stdint.h>

#include "msentity/c/msentity_c.h" /* status codes and msentity_value */

#if defined(_WIN32)
#if defined(MSSIM_C_BUILD)
#define MSSIM_C_API __declspec(dllexport)
#else
#define MSSIM_C_API __declspec(dllimport)
#endif
#else
#define MSSIM_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MSSIM_METHOD_COSINE 0
#define MSSIM_METHOD_REVERSE_COSINE 1

typedef struct mssim_options {
  int32_t method;            /* MSSIM_METHOD_* */
  int32_t threads;           /* 0 = all hardware threads */
  double bin_width;          /* > 0, default 0.01 */
  double intensity_exponent; /* > 0, default 1.0 */
} mssim_options;

/* (completed pairs, total pairs, user data) */
typedef void (*mssim_progress_fn)(int64_t done, int64_t total, void* user_data);

typedef struct mssim_result mssim_result;

/* ---- library ---------------------------------------------------------------- */
MSSIM_C_API const char* mssim_version(void);
MSSIM_C_API const char* mssim_last_error(void);
MSSIM_C_API void mssim_string_free(char* text);
MSSIM_C_API void mssim_bytes_free(uint8_t* data);
MSSIM_C_API void mssim_values_clear(msentity_value* values, int64_t n);
MSSIM_C_API void mssim_default_options(mssim_options* options);

/* ---- calculations ----------------------------------------------------------------- */
/* Scores of ds1[index1[k]] vs ds2[index2[k]], k < n; out has n floats. */
MSSIM_C_API int32_t mssim_pair_scores(const uint8_t* ds1, int64_t ds1_size, const int64_t* index1,
                                      const uint8_t* ds2, int64_t ds2_size, const int64_t* index2,
                                      int64_t n, const mssim_options* options, float* out);
/* Row-major size(ds1) × size(ds2) matrix; n must equal that product. */
MSSIM_C_API int32_t mssim_all_pairs(const uint8_t* ds1, int64_t ds1_size, const uint8_t* ds2,
                                    int64_t ds2_size, const mssim_options* options, float* out,
                                    int64_t n);
/* Library search with matched data and calculation metadata. Sources (file
 * names for the metadata) and progress may be NULL. */
MSSIM_C_API int32_t mssim_library_search(const uint8_t* query, int64_t query_size,
                                         const uint8_t* reference, int64_t reference_size,
                                         double threshold, const mssim_options* options,
                                         int32_t include_matched_data, const char* query_source,
                                         const char* reference_source, mssim_progress_fn progress,
                                         void* user_data, mssim_result** out);
/* Pair spectra by equal metadata keys. */
MSSIM_C_API int32_t mssim_similarity_by_key(const uint8_t* ds1, int64_t ds1_size,
                                            const uint8_t* ds2, int64_t ds2_size,
                                            const char* key1, const char* key2,
                                            const mssim_options* options,
                                            int32_t include_matched_data, const char* source1,
                                            const char* source2, mssim_result** out);

/* ---- results (.mssim) ------------------------------------------------------------------- */
MSSIM_C_API int32_t mssim_load(const char* path, mssim_result** out);
MSSIM_C_API int32_t mssim_save(const mssim_result* result, const char* path);
MSSIM_C_API void mssim_free(mssim_result* result);

MSSIM_C_API int64_t mssim_size(const mssim_result* result);
MSSIM_C_API int64_t mssim_n_columns(const mssim_result* result);
MSSIM_C_API int32_t mssim_column_name(const mssim_result* result, int64_t index, char** out);
/* out has mssim_size() values. */
MSSIM_C_API int32_t mssim_get_column(const mssim_result* result, const char* column,
                                     msentity_value* out, int64_t n);
MSSIM_C_API int32_t mssim_get_metadata_json(const mssim_result* result, char** out);
MSSIM_C_API int32_t mssim_set_metadata_json(mssim_result* result, const char* json);

MSSIM_C_API int32_t mssim_has_matched_data(const mssim_result* result);
/* side: 1 (query / first) or 2 (reference / second). */
MSSIM_C_API int32_t mssim_matched_dataset(const mssim_result* result, int32_t side,
                                          uint8_t** out, int64_t* size);
MSSIM_C_API int64_t mssim_matched_size(const mssim_result* result, int32_t side);
MSSIM_C_API int32_t mssim_source_indices(const mssim_result* result, int32_t side, int64_t* out,
                                         int64_t n);
/* Rows of the matched datasets for a result row. */
MSSIM_C_API int32_t mssim_match_positions(const mssim_result* result, int64_t row,
                                          int64_t* data_index1, int64_t* data_index2);

MSSIM_C_API int32_t mssim_filter(const mssim_result* result, const uint8_t* mask, int64_t n,
                                 mssim_result** out);
MSSIM_C_API int32_t mssim_sort_by(const mssim_result* result, const char* const* columns,
                                  const int32_t* ascending, int64_t n, mssim_result** out);
/* out[7] = count, mean, q1, median, q3, min, max. */
MSSIM_C_API int32_t mssim_describe_scores(const mssim_result* result, double* out);
/* counts has bins entries, edges bins + 1. */
MSSIM_C_API int32_t mssim_histogram(const mssim_result* result, int32_t bins, int64_t* counts,
                                    double* edges);
MSSIM_C_API int32_t mssim_export_table(const mssim_result* result, const char* path);
/* Result table as an Arrow C stream (struct ArrowArrayStream*). */
MSSIM_C_API int32_t mssim_table_arrow(const mssim_result* result, void* out_stream);

#ifdef __cplusplus
}
#endif

#endif /* MSENTITY_SIMILARITY_C_H */
