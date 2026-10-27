package io.github.motohiroogawa.msentity.similarity;

/**
 * An error reported by the msentity-similarity library. The subclass tells the kind; the status
 * codes are those of msentity (MSENTITY_ERR_*).
 */
public class SimilarityException extends RuntimeException {
  private final int status;

  SimilarityException(int status, String message) {
    super(message);
    this.status = status;
  }

  /** The C API status code (MSENTITY_ERR_*). */
  public int status() {
    return status;
  }

  static SimilarityException of(int status, String message) {
    return switch (status) {
      case 1 -> new InvalidArgument(message);
      case 2 -> new PositionOutOfRange(message);
      case 3 -> new ColumnNotFound(message);
      case 4 -> new Format(message);
      case 5 -> new FileAccess(message);
      default -> new SimilarityException(status, message);
    };
  }

  /** Invalid argument or invalid data. */
  public static final class InvalidArgument extends SimilarityException {
    InvalidArgument(String message) {
      super(1, message);
    }
  }

  /** Spectrum or row position out of range. */
  public static final class PositionOutOfRange extends SimilarityException {
    PositionOutOfRange(String message) {
      super(2, message);
    }
  }

  /** Unknown column. */
  public static final class ColumnNotFound extends SimilarityException {
    ColumnNotFound(String message) {
      super(3, message);
    }
  }

  /** Malformed or unreadable file or data. */
  public static final class Format extends SimilarityException {
    Format(String message) {
      super(4, message);
    }
  }

  /** File cannot be opened or written. */
  public static final class FileAccess extends SimilarityException {
    FileAccess(String message) {
      super(5, message);
    }
  }
}
