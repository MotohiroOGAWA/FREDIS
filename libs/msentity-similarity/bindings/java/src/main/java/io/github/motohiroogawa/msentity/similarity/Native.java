package io.github.motohiroogawa.msentity.similarity;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Downcall handles for the msentity-similarity C API
 * (include/msentity/similarity/c/msentity_similarity_c.h).
 */
final class Native {
  private Native() {}

  // ---- types --------------------------------------------------------------------------------
  static final int TYPE_NULL = 0;
  static final int TYPE_BOOL = 1;
  static final int TYPE_INT = 2;
  static final int TYPE_FLOAT = 3;
  static final int TYPE_STRING = 4;

  /** struct msentity_value */
  static final StructLayout VALUE =
      MemoryLayout.structLayout(
          JAVA_INT.withName("type"),
          JAVA_INT.withName("bool_value"),
          JAVA_LONG.withName("int_value"),
          JAVA_DOUBLE.withName("float_value"),
          ADDRESS.withName("string_value"));

  static final VarHandle VALUE_TYPE = VALUE.varHandle(MemoryLayout.PathElement.groupElement("type"));
  static final VarHandle VALUE_BOOL =
      VALUE.varHandle(MemoryLayout.PathElement.groupElement("bool_value"));
  static final VarHandle VALUE_INT =
      VALUE.varHandle(MemoryLayout.PathElement.groupElement("int_value"));
  static final VarHandle VALUE_FLOAT =
      VALUE.varHandle(MemoryLayout.PathElement.groupElement("float_value"));
  static final VarHandle VALUE_STRING =
      VALUE.varHandle(MemoryLayout.PathElement.groupElement("string_value"));

  /** struct mssim_options */
  static final StructLayout OPTIONS =
      MemoryLayout.structLayout(
          JAVA_INT.withName("method"),
          JAVA_INT.withName("threads"),
          JAVA_DOUBLE.withName("bin_width"),
          JAVA_DOUBLE.withName("intensity_exponent"));

  // ---- loading ------------------------------------------------------------------------------
  static final Linker LINKER = Linker.nativeLinker();
  private static final String LIBRARY = "msentity_similarity_c";
  private static final SymbolLookup LIB = lookup();

  /**
   * Find the C library: -Dmssim.library, then MSSIM_C_LIBRARY, then the copy bundled in this JAR
   * for the current platform, then the system library path.
   */
  private static SymbolLookup lookup() {
    String explicit = System.getProperty("mssim.library");
    if (explicit == null || explicit.isEmpty()) explicit = System.getenv("MSSIM_C_LIBRARY");
    if (explicit != null && !explicit.isEmpty()) {
      Path path = Path.of(explicit);
      if (!Files.exists(path)) {
        throw new UnsatisfiedLinkError("msentity-similarity C library not found: " + path);
      }
      return SymbolLookup.libraryLookup(path, Arena.global());
    }
    Path bundled = extractBundledLibrary();
    if (bundled != null) return SymbolLookup.libraryLookup(bundled, Arena.global());
    return SymbolLookup.libraryLookup(System.mapLibraryName(LIBRARY), Arena.global());
  }

  /** Platform name used for the bundled libraries: linux-x64, win-x64, osx-arm64, … */
  static String platform() {
    String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
    String osName = os.contains("win") ? "win" : (os.contains("mac") || os.contains("darwin")) ? "osx" : "linux";
    String archName =
        switch (arch) {
          case "amd64", "x86_64" -> "x64";
          case "aarch64", "arm64" -> "arm64";
          default -> arch;
        };
    return osName + "-" + archName;
  }

  /** Copy the library bundled for this platform to a temporary directory, or return null. */
  private static Path extractBundledLibrary() {
    String name = System.mapLibraryName(LIBRARY);
    String resource =
        "/io/github/motohiroogawa/msentity/similarity/native/" + platform() + "/" + name;
    try (InputStream in = Native.class.getResourceAsStream(resource)) {
      if (in == null) return null;
      Path directory = Files.createTempDirectory("msentity-similarity-");
      Path library = directory.resolve(name);
      Files.copy(in, library, StandardCopyOption.REPLACE_EXISTING);
      library.toFile().deleteOnExit();
      directory.toFile().deleteOnExit();
      return library;
    } catch (IOException e) {
      throw new UncheckedIOException("cannot extract the bundled msentity-similarity library", e);
    }
  }

  private static MethodHandle fn(String name, MemoryLayout result, MemoryLayout... args) {
    FunctionDescriptor descriptor =
        result == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(result, args);
    return LINKER.downcallHandle(
        LIB.find(name).orElseThrow(() -> new UnsatisfiedLinkError("missing symbol " + name)),
        descriptor);
  }

  private static final MemoryLayout P = ADDRESS;
  private static final MemoryLayout I32 = JAVA_INT;
  private static final MemoryLayout I64 = JAVA_LONG;
  private static final MemoryLayout F64 = JAVA_DOUBLE;

  /** void (*mssim_progress_fn)(int64_t done, int64_t total, void* user_data) */
  static final FunctionDescriptor PROGRESS = FunctionDescriptor.ofVoid(I64, I64, P);

  // ---- functions ---------------------------------------------------------------------------------
  static final MethodHandle version = fn("mssim_version", P);
  static final MethodHandle lastError = fn("mssim_last_error", P);
  static final MethodHandle stringFree = fn("mssim_string_free", null, P);
  static final MethodHandle bytesFree = fn("mssim_bytes_free", null, P);
  static final MethodHandle valuesClear = fn("mssim_values_clear", null, P, I64);

  static final MethodHandle pairScores = fn("mssim_pair_scores", I32, P, I64, P, P, I64, P, I64, P, P);
  static final MethodHandle allPairs = fn("mssim_all_pairs", I32, P, I64, P, I64, P, P, I64);
  static final MethodHandle librarySearch =
      fn("mssim_library_search", I32, P, I64, P, I64, F64, P, I32, P, P, P, P, P);
  static final MethodHandle similarityByKey =
      fn("mssim_similarity_by_key", I32, P, I64, P, I64, P, P, P, I32, P, P, P);

  static final MethodHandle load = fn("mssim_load", I32, P, P);
  static final MethodHandle save = fn("mssim_save", I32, P, P);
  static final MethodHandle free = fn("mssim_free", null, P);

  static final MethodHandle size = fn("mssim_size", I64, P);
  static final MethodHandle nColumns = fn("mssim_n_columns", I64, P);
  static final MethodHandle columnName = fn("mssim_column_name", I32, P, I64, P);
  static final MethodHandle getColumn = fn("mssim_get_column", I32, P, P, P, I64);
  static final MethodHandle getMetadataJson = fn("mssim_get_metadata_json", I32, P, P);
  static final MethodHandle setMetadataJson = fn("mssim_set_metadata_json", I32, P, P);

  static final MethodHandle hasMatchedData = fn("mssim_has_matched_data", I32, P);
  static final MethodHandle matchedDataset = fn("mssim_matched_dataset", I32, P, I32, P, P);
  static final MethodHandle matchedSize = fn("mssim_matched_size", I64, P, I32);
  static final MethodHandle sourceIndices = fn("mssim_source_indices", I32, P, I32, P, I64);
  static final MethodHandle matchPositions = fn("mssim_match_positions", I32, P, I64, P, P);

  static final MethodHandle filter = fn("mssim_filter", I32, P, P, I64, P);
  static final MethodHandle sortBy = fn("mssim_sort_by", I32, P, P, P, I64, P);
  static final MethodHandle describeScores = fn("mssim_describe_scores", I32, P, P);
  static final MethodHandle histogram = fn("mssim_histogram", I32, P, I32, P, P);
  static final MethodHandle exportTable = fn("mssim_export_table", I32, P, P);

  // ---- helpers ------------------------------------------------------------------------------------

  /** A native call that may throw Throwable (MethodHandle.invoke). */
  @FunctionalInterface
  interface Call<T> {
    T run() throws Throwable;
  }

  static <T> T call(Call<T> call) {
    try {
      return call.run();
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("native call failed", t);
    }
  }

  /** Throw the exception matching a non-zero status. */
  static void check(int status) {
    if (status == 0) return;
    MemorySegment message = call(() -> (MemorySegment) lastError.invoke());
    throw SimilarityException.of(status, cString(message));
  }

  static String cString(MemorySegment pointer) {
    if (pointer.equals(MemorySegment.NULL)) return "";
    return pointer.reinterpret(Long.MAX_VALUE).getString(0);
  }

  /** Read and free a string returned through a char** out-parameter. */
  static String takeString(MemorySegment slot) {
    MemorySegment pointer = slot.get(ADDRESS, 0);
    try {
      return cString(pointer);
    } finally {
      call(() -> stringFree.invoke(pointer));
    }
  }

  static MemorySegment string(Arena arena, String text) {
    return text == null ? MemorySegment.NULL : arena.allocateFrom(text);
  }
}
