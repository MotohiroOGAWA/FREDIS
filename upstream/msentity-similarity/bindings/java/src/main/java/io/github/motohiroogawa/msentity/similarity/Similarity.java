package io.github.motohiroogawa.msentity.similarity;

import static io.github.motohiroogawa.msentity.similarity.Native.call;
import static io.github.motohiroogawa.msentity.similarity.Native.check;
import static io.github.motohiroogawa.msentity.similarity.Native.string;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.motohiroogawa.msentity.Dataset;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * Similarity calculations on msentity datasets.
 *
 * <pre>{@code
 * try (Dataset query = Dataset.load(Path.of("query.msp"));
 *     Dataset library = Dataset.load(Path.of("library.msp"));
 *     SimilarityDataset hits = Similarity.librarySearch(query, library, 0.8)) {
 *   hits.column("cosine_similarity");
 *   hits.save(Path.of("hits.mssim"));
 * }
 * }</pre>
 *
 * <p>Datasets are handed to the C library as .msds bytes ({@link Dataset#toBytes}).
 */
public final class Similarity {
  private Similarity() {}

  /** Receives (completed pairs, total pairs) during a library search; called from worker threads. */
  @FunctionalInterface
  public interface Progress {
    void report(long done, long total);
  }

  /** Version of the native library. */
  public static String version() {
    return Native.cString(call(() -> (MemorySegment) Native.version.invoke()));
  }

  private static MemorySegment bytes(Arena arena, Dataset dataset) {
    byte[] data = dataset.toBytes();
    return arena.allocateFrom(JAVA_BYTE, data);
  }

  /** Scores of ds1[index1[k]] vs ds2[index2[k]]. */
  public static float[] pairScores(
      Dataset ds1, long[] index1, Dataset ds2, long[] index2, Options options) {
    if (index1.length != index2.length) {
      throw new SimilarityException.InvalidArgument("index1 and index2 must have the same length");
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment a = bytes(arena, ds1);
      MemorySegment b = bytes(arena, ds2);
      MemorySegment i1 = arena.allocateFrom(JAVA_LONG, index1);
      MemorySegment i2 = arena.allocateFrom(JAVA_LONG, index2);
      MemorySegment out = arena.allocate(JAVA_FLOAT, Math.max(1, index1.length));
      MemorySegment opts = options.toNative(arena);
      check(
          call(
              () ->
                  (int)
                      Native.pairScores.invoke(
                          a, a.byteSize(), i1, b, b.byteSize(), i2, (long) index1.length, opts,
                          out)));
      return out.asSlice(0, (long) index1.length * Float.BYTES).toArray(JAVA_FLOAT);
    }
  }

  public static float[] pairScores(Dataset ds1, long[] index1, Dataset ds2, long[] index2) {
    return pairScores(ds1, index1, ds2, index2, Options.defaults());
  }

  /** size(ds1) × size(ds2) score matrix. */
  public static float[][] allPairs(Dataset ds1, Dataset ds2, Options options) {
    int n1 = Math.toIntExact(ds1.size());
    int n2 = Math.toIntExact(ds2.size());
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment a = bytes(arena, ds1);
      MemorySegment b = bytes(arena, ds2);
      long n = (long) n1 * n2;
      MemorySegment out = arena.allocate(JAVA_FLOAT, Math.max(1, n));
      MemorySegment opts = options.toNative(arena);
      check(
          call(
              () -> (int) Native.allPairs.invoke(a, a.byteSize(), b, b.byteSize(), opts, out, n)));
      float[][] matrix = new float[n1][];
      for (int i = 0; i < n1; i++) {
        matrix[i] = out.asSlice((long) i * n2 * Float.BYTES, (long) n2 * Float.BYTES).toArray(JAVA_FLOAT);
      }
      return matrix;
    }
  }

  public static float[][] allPairs(Dataset ds1, Dataset ds2) {
    return allPairs(ds1, ds2, Options.defaults());
  }

  private static final MethodHandle PROGRESS_TARGET;

  static {
    try {
      PROGRESS_TARGET =
          MethodHandles.lookup()
              .findStatic(
                  Similarity.class,
                  "reportProgress",
                  MethodType.methodType(
                      void.class, Progress.class, long.class, long.class, MemorySegment.class));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @SuppressWarnings("unused") // called from native code
  private static void reportProgress(Progress progress, long done, long total, MemorySegment unused) {
    try {
      progress.report(done, total);
    } catch (Throwable t) {
      // An exception must not unwind into native code: report it and continue.
      t.printStackTrace();
    }
  }

  /**
   * All (query, reference) pairs scoring at least {@code threshold}.
   *
   * @param includeMatchedData embed the compared spectra in the result
   * @param querySource recorded in the metadata (for example a file name); may be null
   * @param referenceSource recorded in the metadata; may be null
   * @param progress may be null
   */
  public static SimilarityDataset librarySearch(
      Dataset query,
      Dataset reference,
      double threshold,
      Options options,
      boolean includeMatchedData,
      String querySource,
      String referenceSource,
      Progress progress) {
    // A shared arena: the progress stub is called from worker threads.
    try (Arena arena = Arena.ofShared()) {
      MemorySegment a = bytes(arena, query);
      MemorySegment b = bytes(arena, reference);
      MemorySegment opts = options.toNative(arena);
      MemorySegment q = string(arena, querySource);
      MemorySegment r = string(arena, referenceSource);
      MemorySegment stub =
          progress == null
              ? MemorySegment.NULL
              : Native.LINKER.upcallStub(
                  MethodHandles.insertArguments(PROGRESS_TARGET, 0, progress),
                  Native.PROGRESS,
                  arena);
      MemorySegment out = arena.allocate(ADDRESS);
      check(
          call(
              () ->
                  (int)
                      Native.librarySearch.invoke(
                          a, a.byteSize(), b, b.byteSize(), threshold, opts,
                          includeMatchedData ? 1 : 0, q, r, stub, MemorySegment.NULL, out)));
      return new SimilarityDataset(out.get(ADDRESS, 0));
    }
  }

  /** Library search with embedded spectra and no progress reporting. */
  public static SimilarityDataset librarySearch(
      Dataset query, Dataset reference, double threshold, Options options) {
    return librarySearch(query, reference, threshold, options, true, null, null, null);
  }

  public static SimilarityDataset librarySearch(Dataset query, Dataset reference, double threshold) {
    return librarySearch(query, reference, threshold, Options.defaults());
  }

  /**
   * Scores of the spectrum pairs whose {@code key1} / {@code key2} values are equal (each value
   * must be unique within its dataset).
   */
  public static SimilarityDataset similarityByKey(
      Dataset ds1,
      Dataset ds2,
      String key1,
      String key2,
      Options options,
      boolean includeMatchedData,
      String source1,
      String source2) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment a = bytes(arena, ds1);
      MemorySegment b = bytes(arena, ds2);
      MemorySegment opts = options.toNative(arena);
      MemorySegment k1 = string(arena, key1);
      MemorySegment k2 = string(arena, key2);
      MemorySegment s1 = string(arena, source1);
      MemorySegment s2 = string(arena, source2);
      MemorySegment out = arena.allocate(ADDRESS);
      check(
          call(
              () ->
                  (int)
                      Native.similarityByKey.invoke(
                          a, a.byteSize(), b, b.byteSize(), k1, k2, opts,
                          includeMatchedData ? 1 : 0, s1, s2, out)));
      return new SimilarityDataset(out.get(ADDRESS, 0));
    }
  }

  public static SimilarityDataset similarityByKey(
      Dataset ds1, Dataset ds2, String key1, String key2, Options options) {
    return similarityByKey(ds1, ds2, key1, key2, options, false, null, null);
  }

  public static SimilarityDataset similarityByKey(Dataset ds1, Dataset ds2) {
    return similarityByKey(ds1, ds2, "SpecID", "SpecID", Options.defaults());
  }
}
