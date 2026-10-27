package io.github.motohiroogawa.msentity.similarity;

import static io.github.motohiroogawa.msentity.similarity.Native.call;
import static io.github.motohiroogawa.msentity.similarity.Native.check;
import static io.github.motohiroogawa.msentity.similarity.Native.string;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.motohiroogawa.msentity.Dataset;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A similarity result (one row per spectrum pair), as stored in .mssim files.
 *
 * <p>Columns: key columns (by-key results), {@code index1}, {@code index2} (positions in the input
 * datasets), {@code cosine_similarity} (the score of any method), and {@code data_index1} / {@code
 * data_index2} when the compared spectra are embedded. Cell values are {@code null}, {@link
 * Boolean}, {@link Long}, {@link Double}, or {@link String}.
 *
 * <p>Close results with {@link #close} (or try-with-resources); unclosed results are released
 * when garbage collected.
 */
public final class SimilarityDataset implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();

  private final MemorySegment handle;
  private final Cleaner.Cleanable cleanable;

  SimilarityDataset(MemorySegment handle) {
    this.handle = handle;
    this.cleanable = CLEANER.register(this, () -> call(() -> Native.free.invoke(handle)));
  }

  /** Release the native result now. */
  @Override
  public void close() {
    cleanable.clean();
  }

  @FunctionalInterface
  private interface Creator {
    int create(MemorySegment out) throws Throwable;
  }

  private static SimilarityDataset create(Arena arena, Creator creator) {
    MemorySegment out = arena.allocate(ADDRESS);
    check(call(() -> creator.create(out)));
    return new SimilarityDataset(out.get(ADDRESS, 0));
  }

  /** Read an .mssim file. */
  public static SimilarityDataset load(Path path) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment p = arena.allocateFrom(path.toString());
      return create(arena, out -> (int) Native.load.invoke(p, out));
    }
  }

  /** Write an .mssim file (embedded spectra are reduced to those still referenced). */
  public void save(Path path) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment p = arena.allocateFrom(path.toString());
      check(call(() -> (int) Native.save.invoke(handle, p)));
    }
  }

  /** Write the table only, as .parquet, .csv, or .tsv. */
  public void exportTable(Path path) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment p = arena.allocateFrom(path.toString());
      check(call(() -> (int) Native.exportTable.invoke(handle, p)));
    }
  }

  // ---- table ------------------------------------------------------------------------------------

  /** Number of rows (spectrum pairs). */
  public long size() {
    return call(() -> (long) Native.size.invoke(handle));
  }

  public List<String> columns() {
    long n = call(() -> (long) Native.nColumns.invoke(handle));
    List<String> names = new ArrayList<>();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment slot = arena.allocate(ADDRESS);
      for (long i = 0; i < n; i++) {
        long index = i;
        check(call(() -> (int) Native.columnName.invoke(handle, index, slot)));
        names.add(Native.takeString(slot));
      }
    }
    return names;
  }

  private static Object readValue(MemorySegment value) {
    int type = (int) Native.VALUE_TYPE.get(value, 0L);
    return switch (type) {
      case Native.TYPE_BOOL -> (int) Native.VALUE_BOOL.get(value, 0L) != 0;
      case Native.TYPE_INT -> (long) Native.VALUE_INT.get(value, 0L);
      case Native.TYPE_FLOAT -> (double) Native.VALUE_FLOAT.get(value, 0L);
      case Native.TYPE_STRING -> Native.cString((MemorySegment) Native.VALUE_STRING.get(value, 0L));
      default -> null;
    };
  }

  /** All values of a column. */
  public List<Object> column(String name) {
    long n = size();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment column = string(arena, name);
      MemorySegment array = arena.allocate(Native.VALUE, Math.max(1, n));
      check(call(() -> (int) Native.getColumn.invoke(handle, column, array, n)));
      List<Object> values = new ArrayList<>((int) n);
      try {
        long stride = Native.VALUE.byteSize();
        for (long i = 0; i < n; i++) values.add(readValue(array.asSlice(i * stride, stride)));
      } finally {
        call(() -> Native.valuesClear.invoke(array, n));
      }
      return values;
    }
  }

  // ---- metadata ---------------------------------------------------------------------------------

  /** Calculation metadata (parameters, sources, created_at, …) as a JSON object. */
  public String metadataJson() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment slot = arena.allocate(ADDRESS);
      check(call(() -> (int) Native.getMetadataJson.invoke(handle, slot)));
      return Native.takeString(slot);
    }
  }

  /** Replace the metadata; {@code json} must be a JSON object. */
  public void setMetadataJson(String json) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment text = string(arena, json);
      check(call(() -> (int) Native.setMetadataJson.invoke(handle, text)));
    }
  }

  // ---- matched data -----------------------------------------------------------------------------

  public boolean hasMatchedData() {
    return call(() -> (int) Native.hasMatchedData.invoke(handle)) != 0;
  }

  /**
   * The embedded spectra of one side (1 = query / first, 2 = reference / second); only those the
   * rows refer to.
   */
  public Dataset matchedDataset(int side) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment pointer = arena.allocate(ADDRESS);
      MemorySegment size = arena.allocate(JAVA_LONG);
      check(call(() -> (int) Native.matchedDataset.invoke(handle, side, pointer, size)));
      MemorySegment data = pointer.get(ADDRESS, 0);
      try {
        return Dataset.fromBytes(data.reinterpret(size.get(JAVA_LONG, 0)).toArray(JAVA_BYTE));
      } finally {
        call(() -> Native.bytesFree.invoke(data));
      }
    }
  }

  /** For each embedded spectrum of {@code side}, its position in the original input dataset. */
  public long[] sourceIndices(int side) {
    long n = call(() -> (long) Native.matchedSize.invoke(handle, side));
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG, Math.max(1, n));
      check(call(() -> (int) Native.sourceIndices.invoke(handle, side, out, n)));
      return out.asSlice(0, n * Long.BYTES).toArray(JAVA_LONG);
    }
  }

  /** The two spectra compared in one row, as one-spectrum datasets. */
  public record Match(Dataset first, Dataset second) implements AutoCloseable {
    @Override
    public void close() {
      first.close();
      second.close();
    }
  }

  /** The two spectra compared in {@code row}. */
  public Match matchRecords(long row) {
    long[] positions = new long[2];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment d1 = arena.allocate(JAVA_LONG);
      MemorySegment d2 = arena.allocate(JAVA_LONG);
      check(call(() -> (int) Native.matchPositions.invoke(handle, row, d1, d2)));
      positions[0] = d1.get(JAVA_LONG, 0);
      positions[1] = d2.get(JAVA_LONG, 0);
    }
    try (Dataset first = matchedDataset(1);
        Dataset second = matchedDataset(2)) {
      return new Match(first.select(positions[0]), second.select(positions[1]));
    }
  }

  // ---- selection --------------------------------------------------------------------------------

  /** Rows where {@code mask} is true. */
  public SimilarityDataset filter(boolean[] mask) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment bytes = arena.allocate(JAVA_BYTE, Math.max(1, mask.length));
      for (int i = 0; i < mask.length; i++) bytes.set(JAVA_BYTE, i, (byte) (mask[i] ? 1 : 0));
      return create(arena, out -> (int) Native.filter.invoke(handle, bytes, (long) mask.length, out));
    }
  }

  /** Stable sort by several columns; missing values last. */
  public SimilarityDataset sortBy(List<String> columns, boolean[] ascending) {
    if (ascending.length != columns.size()) {
      throw new SimilarityException.InvalidArgument("ascending must have one entry per column");
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment names = arena.allocate(ADDRESS, Math.max(1, columns.size()));
      MemorySegment orders = arena.allocate(JAVA_INT, Math.max(1, columns.size()));
      for (int i = 0; i < columns.size(); i++) {
        names.setAtIndex(ADDRESS, i, arena.allocateFrom(columns.get(i)));
        orders.setAtIndex(JAVA_INT, i, ascending[i] ? 1 : 0);
      }
      return create(
          arena,
          out -> (int) Native.sortBy.invoke(handle, names, orders, (long) columns.size(), out));
    }
  }

  public SimilarityDataset sortBy(String column, boolean ascending) {
    return sortBy(List.of(column), new boolean[] {ascending});
  }

  // ---- statistics -------------------------------------------------------------------------------

  /** Summary of the score column; NaN statistics when empty. */
  public record ScoreSummary(
      long count, double mean, double q1, double median, double q3, double min, double max) {}

  public ScoreSummary describeScores() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_DOUBLE, 7);
      check(call(() -> (int) Native.describeScores.invoke(handle, out)));
      double[] v = out.toArray(JAVA_DOUBLE);
      return new ScoreSummary((long) v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }
  }

  /** Score counts over [0, 1]; {@code edges} has {@code counts.length + 1} entries. */
  public record Histogram(long[] counts, double[] edges) {}

  public Histogram histogram(int bins) {
    if (bins < 1 || bins > 200) {
      throw new SimilarityException.InvalidArgument("bins must be between 1 and 200");
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment counts = arena.allocate(JAVA_LONG, bins);
      MemorySegment edges = arena.allocate(JAVA_DOUBLE, bins + 1);
      check(call(() -> (int) Native.histogram.invoke(handle, bins, counts, edges)));
      return new Histogram(counts.toArray(JAVA_LONG), edges.toArray(JAVA_DOUBLE));
    }
  }

  @Override
  public String toString() {
    return "SimilarityDataset(rows=" + size() + ", columns=" + columns() + ", matched_data="
        + (hasMatchedData() ? "embedded" : "lightweight") + ")";
  }
}
