package io.github.motohiroogawa.msentity;

import static io.github.motohiroogawa.msentity.Native.call;
import static io.github.motohiroogawa.msentity.Native.check;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Spectra with their metadata and peaks.
 *
 * <p>Cell values are {@code null}, {@link Boolean}, {@link Long}, {@link Double}, or {@link
 * String}. {@link #select}, {@link #slice}, {@link #filter}, {@link #sortBy}, and {@link #fullView}
 * return views that share data with this dataset; {@link #copy} makes an independent one.
 *
 * <p>Close datasets with {@link #close} (or try-with-resources); unclosed datasets are released
 * when garbage collected.
 */
public final class Dataset implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();

  private final MemorySegment handle;
  private final Cleaner.Cleanable cleanable;

  private Dataset(MemorySegment handle) {
    this.handle = handle;
    this.cleanable = CLEANER.register(this, () -> call(() -> Native.free.invoke(handle)));
  }

  /** Release the native dataset now. Views created from it stay valid. */
  @Override
  public void close() {
    cleanable.clean();
  }

  /** Version of the native library. */
  public static String version() {
    return Native.cString(call(() -> (MemorySegment) Native.version.invoke()));
  }

  // ---- creating ---------------------------------------------------------------------------------

  @FunctionalInterface
  private interface Creator {
    int create(MemorySegment out) throws Throwable;
  }

  private static Dataset create(Arena arena, Creator creator) {
    MemorySegment out = arena.allocate(ADDRESS);
    check(call(() -> creator.create(out)));
    return new Dataset(out.get(ADDRESS, 0));
  }

  private static MemorySegment string(Arena arena, String text) {
    return text == null ? MemorySegment.NULL : arena.allocateFrom(text);
  }

  /** Load MSP, MGF, MSDS, TSV, or CSV; the format comes from the extension. */
  public static Dataset load(Path path) {
    return load(path, null, null, false);
  }

  /**
   * Load a file.
   *
   * @param fileType "msp", "mgf", "msds", "tsv", "csv", or null for the extension
   * @param specIdPrefix when not null, add a SpecID column (prefix + number)
   * @param inferTypes convert numeric TSV/CSV columns to numbers
   */
  public static Dataset load(Path path, String fileType, String specIdPrefix, boolean inferTypes) {
    try (Arena arena = Arena.ofConfined()) {
      return create(
          arena,
          out ->
              (int)
                  Native.load.invoke(
                      string(arena, path.toString()),
                      string(arena, fileType),
                      string(arena, specIdPrefix),
                      inferTypes ? 1 : 0,
                      out));
    }
  }

  /** Parse text in "msp", "mgf", "tsv", or "csv" format. */
  public static Dataset readText(String text, String format, boolean inferTypes) {
    try (Arena arena = Arena.ofConfined()) {
      return create(
          arena,
          out ->
              (int)
                  Native.readText.invoke(
                      string(arena, text), string(arena, format), inferTypes ? 1 : 0, out));
    }
  }

  /**
   * Build a dataset without metadata.
   *
   * @param spectra one array per spectrum, each {@code {{mz, intensity}, ...}}
   */
  public static Dataset fromPeaks(double[][][] spectra) {
    long total = 0;
    for (double[][] spectrum : spectra) total += spectrum.length;
    double[] data = new double[Math.toIntExact(total * 2)];
    long[] offsets = new long[spectra.length + 1];
    int k = 0;
    for (int s = 0; s < spectra.length; s++) {
      for (double[] peak : spectra[s]) {
        data[k++] = peak[0];
        data[k++] = peak[1];
      }
      offsets[s + 1] = k / 2;
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment dataSegment = arena.allocateFrom(JAVA_DOUBLE, data);
      MemorySegment offsetSegment = arena.allocateFrom(JAVA_LONG, offsets);
      long peaks = total;
      return create(
          arena,
          out ->
              (int)
                  Native.fromPeaks.invoke(
                      dataSegment, peaks, offsetSegment, (long) offsets.length, out));
    }
  }

  /** Dataset from the bytes of an .msds file (see {@link #toBytes}). */
  public static Dataset fromBytes(byte[] data) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = arena.allocateFrom(JAVA_BYTE, data);
      return create(
          arena, out -> (int) Native.fromMsdsBytes.invoke(segment, (long) data.length, out));
    }
  }

  /**
   * The visible data as the bytes of an .msds file. Other msentity libraries (such as
   * msentity-similarity) exchange datasets this way.
   */
  public byte[] toBytes() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment pointer = arena.allocate(ADDRESS);
      MemorySegment size = arena.allocate(JAVA_LONG);
      check(call(() -> (int) Native.toMsdsBytes.invoke(handle, pointer, size)));
      MemorySegment data = pointer.get(ADDRESS, 0);
      try {
        return data.reinterpret(size.get(JAVA_LONG, 0)).toArray(JAVA_BYTE);
      } finally {
        call(() -> Native.bytesFree.invoke(data));
      }
    }
  }

  /** Join datasets; columns are the union and missing cells are null. */
  public static Dataset concat(Dataset... datasets) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment handles = arena.allocate(ADDRESS, datasets.length);
      for (int i = 0; i < datasets.length; i++) handles.setAtIndex(ADDRESS, i, datasets[i].handle);
      return create(
          arena, out -> (int) Native.concat.invoke(handles, (long) datasets.length, out));
    }
  }

  // ---- shape and columns -----------------------------------------------------------------------------

  /** Number of spectra. */
  public long size() {
    return call(() -> (long) Native.size.invoke(handle));
  }

  public long nPeaksTotal() {
    return call(() -> (long) Native.nPeaksTotal.invoke(handle));
  }

  /** Visible metadata columns in display order. */
  public List<String> columns() {
    long n = call(() -> (long) Native.nColumns.invoke(handle));
    List<String> names = new ArrayList<>();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(ADDRESS);
      for (long i = 0; i < n; i++) {
        long index = i;
        check(call(() -> (int) Native.columnName.invoke(handle, index, out)));
        names.add(Native.takeString(out));
      }
    }
    return Collections.unmodifiableList(names);
  }

  /** Choose the visible columns and their order. */
  public void setColumns(List<String> names) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment array = arena.allocate(ADDRESS, names.size());
      for (int i = 0; i < names.size(); i++) array.setAtIndex(ADDRESS, i, string(arena, names.get(i)));
      check(call(() -> (int) Native.setColumns.invoke(handle, array, (long) names.size())));
    }
  }

  public boolean hasColumn(String name) {
    try (Arena arena = Arena.ofConfined()) {
      return call(() -> (int) Native.hasColumn.invoke(handle, string(arena, name))) != 0;
    }
  }

  // ---- values ---------------------------------------------------------------------------------------------

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

  private static void writeValue(Arena arena, MemorySegment value, Object item) {
    value.fill((byte) 0);
    if (item == null || (item instanceof Double d && d.isNaN())) {
      Native.VALUE_TYPE.set(value, 0L, Native.TYPE_NULL);
    } else if (item instanceof Boolean b) {
      Native.VALUE_TYPE.set(value, 0L, Native.TYPE_BOOL);
      Native.VALUE_BOOL.set(value, 0L, b ? 1 : 0);
    } else if (item instanceof Long || item instanceof Integer || item instanceof Short
        || item instanceof Byte) {
      Native.VALUE_TYPE.set(value, 0L, Native.TYPE_INT);
      Native.VALUE_INT.set(value, 0L, ((Number) item).longValue());
    } else if (item instanceof Double || item instanceof Float) {
      Native.VALUE_TYPE.set(value, 0L, Native.TYPE_FLOAT);
      Native.VALUE_FLOAT.set(value, 0L, ((Number) item).doubleValue());
    } else if (item instanceof String s) {
      Native.VALUE_TYPE.set(value, 0L, Native.TYPE_STRING);
      Native.VALUE_STRING.set(value, 0L, arena.allocateFrom(s));
    } else {
      throw new IllegalArgumentException("unsupported value type: " + item.getClass().getName());
    }
  }

  private static MemorySegment valueArray(Arena arena, List<?> items) {
    MemorySegment array = arena.allocate(Native.VALUE, Math.max(1, items.size()));
    for (int i = 0; i < items.size(); i++) {
      writeValue(arena, array.asSlice(i * Native.VALUE.byteSize(), Native.VALUE.byteSize()), items.get(i));
    }
    return array;
  }

  @FunctionalInterface
  private interface Fill {
    int fill(MemorySegment out, long n) throws Throwable;
  }

  private static List<Object> readValues(long n, Fill fill) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment array = arena.allocate(Native.VALUE, Math.max(1, n));
      check(call(() -> fill.fill(array, n)));
      List<Object> values = new ArrayList<>((int) n);
      try {
        for (long i = 0; i < n; i++) {
          values.add(readValue(array.asSlice(i * Native.VALUE.byteSize(), Native.VALUE.byteSize())));
        }
      } finally {
        call(() -> Native.valuesClear.invoke(array, n));
      }
      return values;
    }
  }

  /** One cell. */
  public Object value(long position, String column) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(Native.VALUE);
      check(call(() -> (int) Native.getValue.invoke(handle, position, string(arena, column), out)));
      try {
        return readValue(out);
      } finally {
        call(() -> Native.valuesClear.invoke(out, 1L));
      }
    }
  }

  /** All values of a column, one per spectrum. */
  public List<Object> column(String name) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment column = string(arena, name);
      return readValues(size(), (out, n) -> (int) Native.getColumn.invoke(handle, column, out, n));
    }
  }

  /** Set a column from one value per spectrum. */
  public void setColumn(String name, List<?> values) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment array = valueArray(arena, values);
      check(
          call(
              () ->
                  (int)
                      Native.setColumn.invoke(
                          handle, string(arena, name), array, (long) values.size())));
    }
  }

  /** Set a column to the same value for every spectrum. */
  public void fillColumn(String name, Object value) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment single = arena.allocate(Native.VALUE);
      writeValue(arena, single, value);
      check(call(() -> (int) Native.setColumnScalar.invoke(handle, string(arena, name), single)));
    }
  }

  /** Set one cell. */
  public void setValue(long position, String column, Object value) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment single = arena.allocate(Native.VALUE);
      writeValue(arena, single, value);
      check(
          call(() -> (int) Native.setValue.invoke(handle, position, string(arena, column), single)));
    }
  }

  // ---- views ---------------------------------------------------------------------------------------------

  /** Spectra at the given positions, in that order (negative = from the end). */
  public Dataset select(long... positions) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment array = arena.allocateFrom(JAVA_LONG, positions);
      return create(
          arena, out -> (int) Native.select.invoke(handle, array, (long) positions.length, out));
    }
  }

  /** Spectra start (inclusive) to stop (exclusive). */
  public Dataset slice(long start, long stop) {
    try (Arena arena = Arena.ofConfined()) {
      return create(arena, out -> (int) Native.slice.invoke(handle, start, stop, out));
    }
  }

  /** Spectra whose mask entry is true (one entry per spectrum). */
  public Dataset filter(boolean[] mask) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment array = arena.allocate(JAVA_BYTE, Math.max(1, mask.length));
      for (int i = 0; i < mask.length; i++) array.set(JAVA_BYTE, i, (byte) (mask[i] ? 1 : 0));
      return create(
          arena, out -> (int) Native.filter.invoke(handle, array, (long) mask.length, out));
    }
  }

  /** Stable sort by a column; missing values last. */
  public Dataset sortBy(String column, boolean ascending) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment name = string(arena, column);
      return create(arena, out -> (int) Native.sortBy.invoke(handle, name, ascending ? 1 : 0, out));
    }
  }

  /** An independent dataset with the visible spectra and columns. */
  public Dataset copy() {
    try (Arena arena = Arena.ofConfined()) {
      return create(arena, out -> (int) Native.copy.invoke(handle, out));
    }
  }

  /** All stored spectra; with resetColumns, also all columns. */
  public Dataset fullView(boolean resetColumns) {
    try (Arena arena = Arena.ofConfined()) {
      return create(
          arena, out -> (int) Native.fullView.invoke(handle, resetColumns ? 1 : 0, out));
    }
  }

  // ---- peaks -------------------------------------------------------------------------------------------------

  /** Spectrum i has peaks offsets[i] … offsets[i + 1] - 1. */
  public long[] offsets() {
    long n = size() + 1;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG, n);
      check(call(() -> (int) Native.offsets.invoke(handle, out, n)));
      return out.toArray(JAVA_LONG);
    }
  }

  private double[][] peakArrays() {
    long n = nPeaksTotal();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment mz = arena.allocate(JAVA_DOUBLE, Math.max(1, n));
      MemorySegment intensity = arena.allocate(JAVA_DOUBLE, Math.max(1, n));
      check(call(() -> (int) Native.peaks.invoke(handle, mz, intensity, n)));
      return new double[][] {
        mz.asSlice(0, n * 8).toArray(JAVA_DOUBLE), intensity.asSlice(0, n * 8).toArray(JAVA_DOUBLE)
      };
    }
  }

  /** m/z of every peak of every spectrum. */
  public double[] mz() {
    return peakArrays()[0];
  }

  /** Intensity of every peak of every spectrum. */
  public double[] intensity() {
    return peakArrays()[1];
  }

  /** The peaks of one spectrum. */
  public Spectrum spectrum(long position) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment size = arena.allocate(JAVA_LONG);
      check(call(() -> (int) Native.spectrumSize.invoke(handle, position, size)));
      long n = size.get(JAVA_LONG, 0);
      MemorySegment mz = arena.allocate(JAVA_DOUBLE, Math.max(1, n));
      MemorySegment intensity = arena.allocate(JAVA_DOUBLE, Math.max(1, n));
      check(call(() -> (int) Native.spectrumPeaks.invoke(handle, position, mz, intensity, n)));
      return new Spectrum(
          mz.asSlice(0, n * 8).toArray(JAVA_DOUBLE), intensity.asSlice(0, n * 8).toArray(JAVA_DOUBLE));
    }
  }

  /** In place: the highest peak of each spectrum becomes {@code scale}. */
  public void normalize(double scale) {
    check(call(() -> (int) Native.normalize.invoke(handle, scale)));
  }

  /** In place: sort the peaks of each spectrum by m/z. */
  public void sortPeaksByMz(boolean ascending) {
    check(call(() -> (int) Native.sortPeaks.invoke(handle, 0, ascending ? 1 : 0)));
  }

  /** In place: sort the peaks of each spectrum by intensity. */
  public void sortPeaksByIntensity(boolean ascending) {
    check(call(() -> (int) Native.sortPeaks.invoke(handle, 1, ascending ? 1 : 0)));
  }

  // ---- peak metadata -------------------------------------------------------------------------------------------

  /** Visible peak metadata columns. */
  public List<String> peakColumns() {
    long n = call(() -> (long) Native.nPeakColumns.invoke(handle));
    List<String> names = new ArrayList<>();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(ADDRESS);
      for (long i = 0; i < n; i++) {
        long index = i;
        check(call(() -> (int) Native.peakColumnName.invoke(handle, index, out)));
        names.add(Native.takeString(out));
      }
    }
    return Collections.unmodifiableList(names);
  }

  /** Values of a peak column, one per peak. */
  public List<Object> peakColumn(String name) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment column = string(arena, name);
      return readValues(
          nPeaksTotal(), (out, n) -> (int) Native.getPeakColumn.invoke(handle, column, out, n));
    }
  }

  /** Set a peak column from one value per peak. */
  public void setPeakColumn(String name, List<?> values) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment array = valueArray(arena, values);
      check(
          call(
              () ->
                  (int)
                      Native.setPeakColumn.invoke(
                          handle, string(arena, name), array, (long) values.size())));
    }
  }

  // ---- dataset-level metadata ------------------------------------------------------------------------------------

  public String description() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(ADDRESS);
      check(call(() -> (int) Native.getDescription.invoke(handle, out)));
      return Native.takeString(out);
    }
  }

  public void setDescription(String text) {
    try (Arena arena = Arena.ofConfined()) {
      check(call(() -> (int) Native.setDescription.invoke(handle, string(arena, text))));
    }
  }

  /** A copy of the attributes in insertion order. */
  public Map<String, String> attributes() {
    long n = call(() -> (long) Native.nAttributes.invoke(handle));
    Map<String, String> result = new LinkedHashMap<>();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment key = arena.allocate(ADDRESS);
      MemorySegment value = arena.allocate(ADDRESS);
      for (long i = 0; i < n; i++) {
        long index = i;
        check(call(() -> (int) Native.attribute.invoke(handle, index, key, value)));
        result.put(Native.takeString(key), Native.takeString(value));
      }
    }
    return result;
  }

  public void setAttribute(String key, String value) {
    try (Arena arena = Arena.ofConfined()) {
      check(
          call(
              () ->
                  (int)
                      Native.setAttribute.invoke(handle, string(arena, key), string(arena, value))));
    }
  }

  /** Returns true if the key existed. */
  public boolean removeAttribute(String key) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment removed = arena.allocate(JAVA_INT);
      check(call(() -> (int) Native.removeAttribute.invoke(handle, string(arena, key), removed)));
      return removed.get(JAVA_INT, 0) != 0;
    }
  }

  public List<String> tags() {
    long n = call(() -> (long) Native.nTags.invoke(handle));
    List<String> result = new ArrayList<>();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(ADDRESS);
      for (long i = 0; i < n; i++) {
        long index = i;
        check(call(() -> (int) Native.tag.invoke(handle, index, out)));
        result.add(Native.takeString(out));
      }
    }
    return Collections.unmodifiableList(result);
  }

  /** Returns false if the tag was already present. */
  public boolean addTag(String tag) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment added = arena.allocate(JAVA_INT);
      check(call(() -> (int) Native.addTag.invoke(handle, string(arena, tag), added)));
      return added.get(JAVA_INT, 0) != 0;
    }
  }

  public boolean removeTag(String tag) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment removed = arena.allocate(JAVA_INT);
      check(call(() -> (int) Native.removeTag.invoke(handle, string(arena, tag), removed)));
      return removed.get(JAVA_INT, 0) != 0;
    }
  }

  /** Sequential zero-padded IDs (SP01 … SP12); false if the column exists and !overwrite. */
  public boolean setSpecId(String column, String prefix, boolean overwrite, long start) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment assigned = arena.allocate(JAVA_INT);
      check(
          call(
              () ->
                  (int)
                      Native.setSpecId.invoke(
                          handle,
                          string(arena, column),
                          string(arena, prefix),
                          overwrite ? 1 : 0,
                          start,
                          assigned)));
      return assigned.get(JAVA_INT, 0) != 0;
    }
  }

  // ---- saving -------------------------------------------------------------------------------------------------------

  /** Save; the format (MSDS, MSP, MGF, TSV, CSV) comes from the extension. */
  public void save(Path path) {
    save(path, null, true, false);
  }

  /**
   * Save.
   *
   * @param format "msds", "msp", "mgf", "tsv", "csv", or null for the extension
   * @param saveView MSDS: store only the visible data (true) or everything with the view
   * @param append MSDS: add a dataset group instead of replacing the file
   */
  public void save(Path path, String format, boolean saveView, boolean append) {
    try (Arena arena = Arena.ofConfined()) {
      check(
          call(
              () ->
                  (int)
                      Native.save.invoke(
                          handle,
                          string(arena, path.toString()),
                          string(arena, format),
                          saveView ? 1 : 0,
                          append ? 1 : 0)));
    }
  }

  @Override
  public String toString() {
    return "Dataset(n_spectra=" + size() + ", n_peaks=" + nPeaksTotal() + ", columns=" + columns() + ")";
  }

  /** The peaks of one spectrum. */
  public record Spectrum(double[] mz, double[] intensity) {
    public int size() {
      return mz.length;
    }
  }
}
