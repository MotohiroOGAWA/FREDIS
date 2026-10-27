package io.github.motohiroogawa.msentity.similarity;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Score settings. Immutable; change values with the {@code with…} methods:
 *
 * <pre>{@code
 * Options options = Options.defaults().withMethod(Method.REVERSE_COSINE).withBinWidth(0.5);
 * }</pre>
 *
 * @param method cosine or reverse cosine
 * @param binWidth m/z bin width (&gt; 0); peaks are binned by floor(mz / binWidth)
 * @param intensityExponent intensities are raised to this power (&gt; 0) before scoring
 * @param threads worker threads; 0 = all hardware threads
 */
public record Options(Method method, double binWidth, double intensityExponent, int threads) {
  /** The scoring method. */
  public enum Method {
    COSINE,
    REVERSE_COSINE
  }

  /** cosine, bin width 0.01, exponent 1, all threads. */
  public static Options defaults() {
    return new Options(Method.COSINE, 0.01, 1.0, 0);
  }

  public Options withMethod(Method value) {
    return new Options(value, binWidth, intensityExponent, threads);
  }

  public Options withBinWidth(double value) {
    return new Options(method, value, intensityExponent, threads);
  }

  public Options withIntensityExponent(double value) {
    return new Options(method, binWidth, value, threads);
  }

  public Options withThreads(int value) {
    return new Options(method, binWidth, intensityExponent, value);
  }

  /** struct mssim_options in {@code arena}. */
  MemorySegment toNative(Arena arena) {
    MemorySegment out = arena.allocate(Native.OPTIONS);
    out.set(ValueLayout.JAVA_INT, 0, method == Method.REVERSE_COSINE ? 1 : 0);
    out.set(ValueLayout.JAVA_INT, 4, threads);
    out.set(ValueLayout.JAVA_DOUBLE, 8, binWidth);
    out.set(ValueLayout.JAVA_DOUBLE, 16, intensityExponent);
    return out;
  }
}
