/*
 * msentity C API — the stable binary interface used by the Python, Java, and
 * C# bindings (and usable from any language with a C FFI).
 *
 * Conventions
 * -----------
 * - Every function that can fail returns an int32_t status (MSENTITY_OK = 0).
 *   On failure, msentity_last_error() returns a message for the calling
 *   thread, valid until the next msentity call on that thread.
 * - msentity_dataset* is an opaque handle owned by the caller; release it
 *   with msentity_free(). Views (select, slice, filter, sort_by, full_view)
 *   are independent handles that share data with the dataset they came from;
 *   msentity_copy() makes an independent dataset.
 * - Strings are UTF-8. Strings returned through char** are allocated by the
 *   library; release them with msentity_string_free().
 * - Arrays are copied into buffers allocated by the caller; the expected
 *   length is passed in and checked.
 * - Rows ("positions") are 0-based positions in the dataset view.
 */
#ifndef MSENTITY_C_H
#define MSENTITY_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(MSENTITY_C_BUILD)
#define MSENTITY_C_API __declspec(dllexport)
#else
#define MSENTITY_C_API __declspec(dllimport)
#endif
#else
#define MSENTITY_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes ------------------------------------------------------- */
#define MSENTITY_OK 0
#define MSENTITY_ERR_INVALID_ARGUMENT 1 /* bad argument or invalid data */
#define MSENTITY_ERR_OUT_OF_RANGE 2     /* position out of range */
#define MSENTITY_ERR_KEY 3              /* unknown column */
#define MSENTITY_ERR_FORMAT 4           /* malformed or unreadable file */
#define MSENTITY_ERR_IO 5               /* file cannot be opened or written */
#define MSENTITY_ERR_UNKNOWN 99

/* ---- values -------------------------------------------------------------- */
#define MSENTITY_TYPE_NULL 0
#define MSENTITY_TYPE_BOOL 1
#define MSENTITY_TYPE_INT 2
#define MSENTITY_TYPE_FLOAT 3
#define MSENTITY_TYPE_STRING 4

/* One metadata cell. Layout (32 bytes on 64-bit platforms):
 *   int32 type, int32 bool_value, int64 int_value, double float_value,
 *   char* string_value.
 * Values returned by the library own string_value: release them with
 * msentity_values_clear(). Values passed to the library are only read. */
typedef struct msentity_value {
  int32_t type;
  int32_t bool_value;
  int64_t int_value;
  double float_value;
  char* string_value;
} msentity_value;

typedef struct msentity_dataset msentity_dataset;

/* ---- library ------------------------------------------------------------- */
MSENTITY_C_API const char* msentity_version(void);
MSENTITY_C_API const char* msentity_last_error(void);
MSENTITY_C_API void msentity_string_free(char* text);
/* Release the strings of n values (the array itself is the caller's). */
MSENTITY_C_API void msentity_values_clear(msentity_value* values, int64_t n);

/* ---- creating and releasing datasets ------------------------------------- */
/* Load MSP, MGF, MSDS, TSV, or CSV. file_type ("msp", "mgf", "msds", "tsv",
 * "csv") may be NULL to use the extension; spec_id_prefix may be NULL;
 * infer_types (0/1) applies to TSV/CSV. */
MSENTITY_C_API int32_t msentity_load(const char* path, const char* file_type,
                                     const char* spec_id_prefix, int32_t infer_types,
                                     msentity_dataset** out);
/* Parse text in "msp", "mgf", "tsv", or "csv" format. */
MSENTITY_C_API int32_t msentity_read_text(const char* text, const char* format,
                                          int32_t infer_types, msentity_dataset** out);
/* Build a dataset without metadata from peaks: data holds n_peaks
 * (m/z, intensity) pairs; offsets has n_spectra + 1 entries. */
MSENTITY_C_API int32_t msentity_from_peaks(const double* data, int64_t n_peaks,
                                           const int64_t* offsets, int64_t n_offsets,
                                           msentity_dataset** out);
/* Join datasets (columns: union; missing cells are null). */
MSENTITY_C_API int32_t msentity_concat(const msentity_dataset* const* datasets, int64_t n,
                                       msentity_dataset** out);
MSENTITY_C_API void msentity_free(msentity_dataset* dataset);

/* ---- saving --------------------------------------------------------------- */
/* format: "msds", "msp", "mgf", "tsv", "csv", or NULL (from the extension).
 * save_view/append apply to MSDS (see MsdsWriteOptions). */
MSENTITY_C_API int32_t msentity_save(const msentity_dataset* dataset, const char* path,
                                     const char* format, int32_t save_view, int32_t append);

/* ---- shape and columns ------------------------------------------------------ */
MSENTITY_C_API int64_t msentity_size(const msentity_dataset* dataset);
MSENTITY_C_API int64_t msentity_n_peaks_total(const msentity_dataset* dataset);
MSENTITY_C_API int64_t msentity_n_columns(const msentity_dataset* dataset);
MSENTITY_C_API int32_t msentity_column_name(const msentity_dataset* dataset, int64_t index,
                                            char** out);
MSENTITY_C_API int32_t msentity_has_column(const msentity_dataset* dataset, const char* name);
/* Visible columns and their order; every name must exist. */
MSENTITY_C_API int32_t msentity_set_columns(msentity_dataset* dataset, const char* const* names,
                                            int64_t n);

/* ---- metadata values ----------------------------------------------------------- */
MSENTITY_C_API int32_t msentity_get_value(const msentity_dataset* dataset, int64_t position,
                                          const char* column, msentity_value* out);
/* out must have msentity_size() elements. */
MSENTITY_C_API int32_t msentity_get_column(const msentity_dataset* dataset, const char* column,
                                           msentity_value* out, int64_t n);
/* values must have msentity_size() elements. */
MSENTITY_C_API int32_t msentity_set_column(msentity_dataset* dataset, const char* column,
                                           const msentity_value* values, int64_t n);
MSENTITY_C_API int32_t msentity_set_column_scalar(msentity_dataset* dataset, const char* column,
                                                  const msentity_value* value);
MSENTITY_C_API int32_t msentity_set_value(msentity_dataset* dataset, int64_t position,
                                          const char* column, const msentity_value* value);

/* ---- views ----------------------------------------------------------------------- */
MSENTITY_C_API int32_t msentity_select(const msentity_dataset* dataset, const int64_t* positions,
                                       int64_t n, msentity_dataset** out);
MSENTITY_C_API int32_t msentity_slice(const msentity_dataset* dataset, int64_t start,
                                      int64_t stop, msentity_dataset** out);
/* mask has msentity_size() bytes (non-zero = keep). */
MSENTITY_C_API int32_t msentity_filter(const msentity_dataset* dataset, const uint8_t* mask,
                                       int64_t n, msentity_dataset** out);
MSENTITY_C_API int32_t msentity_sort_by(const msentity_dataset* dataset, const char* column,
                                        int32_t ascending, msentity_dataset** out);
MSENTITY_C_API int32_t msentity_copy(const msentity_dataset* dataset, msentity_dataset** out);
MSENTITY_C_API int32_t msentity_full_view(const msentity_dataset* dataset,
                                          int32_t reset_columns, msentity_dataset** out);

/* ---- peaks ---------------------------------------------------------------------------- */
/* out has msentity_size() + 1 elements. */
MSENTITY_C_API int32_t msentity_offsets(const msentity_dataset* dataset, int64_t* out,
                                        int64_t n);
/* mz and intensity (either may be NULL) have msentity_n_peaks_total() elements. */
MSENTITY_C_API int32_t msentity_peaks(const msentity_dataset* dataset, double* mz,
                                      double* intensity, int64_t n);
MSENTITY_C_API int32_t msentity_spectrum_size(const msentity_dataset* dataset, int64_t position,
                                              int64_t* out);
MSENTITY_C_API int32_t msentity_spectrum_peaks(const msentity_dataset* dataset,
                                               int64_t position, double* mz,
                                               double* intensity, int64_t n);
/* In place on the shared peaks of the visible spectra. */
MSENTITY_C_API int32_t msentity_normalize(msentity_dataset* dataset, double scale);
/* by: 0 = m/z, 1 = intensity. */
MSENTITY_C_API int32_t msentity_sort_peaks(msentity_dataset* dataset, int32_t by,
                                           int32_t ascending);

/* ---- peak metadata ----------------------------------------------------------------------- */
MSENTITY_C_API int64_t msentity_n_peak_columns(const msentity_dataset* dataset);
MSENTITY_C_API int32_t msentity_peak_column_name(const msentity_dataset* dataset, int64_t index,
                                                 char** out);
/* out has msentity_n_peaks_total() elements. */
MSENTITY_C_API int32_t msentity_get_peak_column(const msentity_dataset* dataset,
                                                const char* column, msentity_value* out,
                                                int64_t n);
MSENTITY_C_API int32_t msentity_set_peak_column(msentity_dataset* dataset, const char* column,
                                                const msentity_value* values, int64_t n);

/* ---- dataset-level metadata ---------------------------------------------------------------- */
MSENTITY_C_API int32_t msentity_get_description(const msentity_dataset* dataset, char** out);
MSENTITY_C_API int32_t msentity_set_description(msentity_dataset* dataset, const char* text);
MSENTITY_C_API int64_t msentity_n_attributes(const msentity_dataset* dataset);
MSENTITY_C_API int32_t msentity_attribute(const msentity_dataset* dataset, int64_t index,
                                          char** key, char** value);
MSENTITY_C_API int32_t msentity_set_attribute(msentity_dataset* dataset, const char* key,
                                              const char* value);
/* *removed = 1 if the key existed. */
MSENTITY_C_API int32_t msentity_remove_attribute(msentity_dataset* dataset, const char* key,
                                                 int32_t* removed);
MSENTITY_C_API int64_t msentity_n_tags(const msentity_dataset* dataset);
MSENTITY_C_API int32_t msentity_tag(const msentity_dataset* dataset, int64_t index, char** out);
MSENTITY_C_API int32_t msentity_add_tag(msentity_dataset* dataset, const char* tag,
                                        int32_t* added);
MSENTITY_C_API int32_t msentity_remove_tag(msentity_dataset* dataset, const char* tag,
                                           int32_t* removed);
/* processing::set_spec_id; *assigned = 0 if the column existed and !overwrite. */
MSENTITY_C_API int32_t msentity_set_spec_id(msentity_dataset* dataset, const char* column,
                                            const char* prefix, int32_t overwrite,
                                            int64_t start, int32_t* assigned);

/* ---- MSDS bytes --------------------------------------------------------------------------------- */
/* A dataset as the bytes of a complete .msds file (materialized view). This
 * is how datasets are exchanged with other msentity libraries, such as
 * msentity-similarity. Release *out with msentity_bytes_free(). */
MSENTITY_C_API int32_t msentity_to_msds_bytes(const msentity_dataset* dataset, uint8_t** out,
                                              int64_t* size);
MSENTITY_C_API int32_t msentity_from_msds_bytes(const uint8_t* data, int64_t size,
                                                msentity_dataset** out);
MSENTITY_C_API void msentity_bytes_free(uint8_t* data);

/* ---- Arrow export ------------------------------------------------------------------------------ */
/* Export the visible spectrum metadata (or peak metadata) as an Arrow C
 * stream. out_stream points to a caller-owned `struct ArrowArrayStream`
 * (Arrow C stream interface); the consumer releases it. */
MSENTITY_C_API int32_t msentity_metadata_arrow(const msentity_dataset* dataset,
                                               void* out_stream);
MSENTITY_C_API int32_t msentity_peak_metadata_arrow(const msentity_dataset* dataset,
                                                    void* out_stream);

#ifdef __cplusplus
}
#endif

#endif /* MSENTITY_C_H */
