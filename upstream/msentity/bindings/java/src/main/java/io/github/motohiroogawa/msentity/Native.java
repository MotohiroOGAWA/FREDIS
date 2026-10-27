package io.github.motohiroogawa.msentity;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/** Downcall handles for the msentity C API (include/msentity/c/msentity_c.h). */
final class Native {
  private Native() {}

  // ---- value types ----------------------------------------------------------------------
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

  // ---- loading ------------------------------------------------------------------------------
  private static final Linker LINKER = Linker.nativeLinker();
  private static final SymbolLookup LIB = lookup();

  /**
   * Find the C library: -Dmsentity.library, then MSENTITY_C_LIBRARY, then the copy bundled in
   * this JAR for the current platform, then the system library path.
   */
  private static SymbolLookup lookup() {
    String explicit = System.getProperty("msentity.library");
    if (explicit == null || explicit.isEmpty()) explicit = System.getenv("MSENTITY_C_LIBRARY");
    if (explicit != null && !explicit.isEmpty()) {
      Path path = Path.of(explicit);
      if (!Files.exists(path)) {
        throw new UnsatisfiedLinkError("msentity C library not found: " + path);
      }
      return SymbolLookup.libraryLookup(path, Arena.global());
    }
    Path bundled = extractBundledLibrary();
    if (bundled != null) return SymbolLookup.libraryLookup(bundled, Arena.global());
    return SymbolLookup.libraryLookup(System.mapLibraryName("msentity_c"), Arena.global());
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
    String name = System.mapLibraryName("msentity_c");
    String resource = "/io/github/motohiroogawa/msentity/native/" + platform() + "/" + name;
    try (InputStream in = Native.class.getResourceAsStream(resource)) {
      if (in == null) return null;
      Path directory = Files.createTempDirectory("msentity-");
      Path library = directory.resolve(name);
      Files.copy(in, library, StandardCopyOption.REPLACE_EXISTING);
      library.toFile().deleteOnExit();
      directory.toFile().deleteOnExit();
      return library;
    } catch (IOException e) {
      throw new UncheckedIOException("cannot extract the bundled msentity library", e);
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

  // ---- functions ---------------------------------------------------------------------------------
  static final MethodHandle version = fn("msentity_version", P);
  static final MethodHandle lastError = fn("msentity_last_error", P);
  static final MethodHandle stringFree = fn("msentity_string_free", null, P);
  static final MethodHandle valuesClear = fn("msentity_values_clear", null, P, I64);

  static final MethodHandle load = fn("msentity_load", I32, P, P, P, I32, P);
  static final MethodHandle readText = fn("msentity_read_text", I32, P, P, I32, P);
  static final MethodHandle fromPeaks = fn("msentity_from_peaks", I32, P, I64, P, I64, P);
  static final MethodHandle concat = fn("msentity_concat", I32, P, I64, P);
  static final MethodHandle free = fn("msentity_free", null, P);
  static final MethodHandle save = fn("msentity_save", I32, P, P, P, I32, I32);

  static final MethodHandle size = fn("msentity_size", I64, P);
  static final MethodHandle nPeaksTotal = fn("msentity_n_peaks_total", I64, P);
  static final MethodHandle nColumns = fn("msentity_n_columns", I64, P);
  static final MethodHandle columnName = fn("msentity_column_name", I32, P, I64, P);
  static final MethodHandle hasColumn = fn("msentity_has_column", I32, P, P);
  static final MethodHandle setColumns = fn("msentity_set_columns", I32, P, P, I64);

  static final MethodHandle getValue = fn("msentity_get_value", I32, P, I64, P, P);
  static final MethodHandle getColumn = fn("msentity_get_column", I32, P, P, P, I64);
  static final MethodHandle setColumn = fn("msentity_set_column", I32, P, P, P, I64);
  static final MethodHandle setColumnScalar = fn("msentity_set_column_scalar", I32, P, P, P);
  static final MethodHandle setValue = fn("msentity_set_value", I32, P, I64, P, P);

  static final MethodHandle select = fn("msentity_select", I32, P, P, I64, P);
  static final MethodHandle slice = fn("msentity_slice", I32, P, I64, I64, P);
  static final MethodHandle filter = fn("msentity_filter", I32, P, P, I64, P);
  static final MethodHandle sortBy = fn("msentity_sort_by", I32, P, P, I32, P);
  static final MethodHandle copy = fn("msentity_copy", I32, P, P);
  static final MethodHandle fullView = fn("msentity_full_view", I32, P, I32, P);

  static final MethodHandle offsets = fn("msentity_offsets", I32, P, P, I64);
  static final MethodHandle peaks = fn("msentity_peaks", I32, P, P, P, I64);
  static final MethodHandle spectrumSize = fn("msentity_spectrum_size", I32, P, I64, P);
  static final MethodHandle spectrumPeaks = fn("msentity_spectrum_peaks", I32, P, I64, P, P, I64);
  static final MethodHandle normalize = fn("msentity_normalize", I32, P, F64);
  static final MethodHandle sortPeaks = fn("msentity_sort_peaks", I32, P, I32, I32);

  static final MethodHandle nPeakColumns = fn("msentity_n_peak_columns", I64, P);
  static final MethodHandle peakColumnName = fn("msentity_peak_column_name", I32, P, I64, P);
  static final MethodHandle getPeakColumn = fn("msentity_get_peak_column", I32, P, P, P, I64);
  static final MethodHandle setPeakColumn = fn("msentity_set_peak_column", I32, P, P, P, I64);

  static final MethodHandle getDescription = fn("msentity_get_description", I32, P, P);
  static final MethodHandle setDescription = fn("msentity_set_description", I32, P, P);
  static final MethodHandle nAttributes = fn("msentity_n_attributes", I64, P);
  static final MethodHandle attribute = fn("msentity_attribute", I32, P, I64, P, P);
  static final MethodHandle setAttribute = fn("msentity_set_attribute", I32, P, P, P);
  static final MethodHandle removeAttribute = fn("msentity_remove_attribute", I32, P, P, P);
  static final MethodHandle nTags = fn("msentity_n_tags", I64, P);
  static final MethodHandle tag = fn("msentity_tag", I32, P, I64, P);
  static final MethodHandle addTag = fn("msentity_add_tag", I32, P, P, P);
  static final MethodHandle removeTag = fn("msentity_remove_tag", I32, P, P, P);
  static final MethodHandle setSpecId = fn("msentity_set_spec_id", I32, P, P, P, I32, I64, P);

  static final MethodHandle toMsdsBytes = fn("msentity_to_msds_bytes", I32, P, P, P);
  static final MethodHandle fromMsdsBytes = fn("msentity_from_msds_bytes", I32, P, I64, P);
  static final MethodHandle bytesFree = fn("msentity_bytes_free", null, P);

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
    throw MsEntityException.of(status, cString(message));
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
}
