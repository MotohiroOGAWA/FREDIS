package io.github.motohiroogawa.msentity;

/** An error reported by the msentity library. The subclass tells the kind. */
public class MsEntityException extends RuntimeException {
  private final int status;

  MsEntityException(int status, String message) {
    super(message);
    this.status = status;
  }

  /** The C API status code (MSENTITY_ERR_*). */
  public int status() {
    return status;
  }

  static MsEntityException of(int status, String message) {
    return switch (status) {
      case 1 -> new InvalidArgument(message);
      case 2 -> new PositionOutOfRange(message);
      case 3 -> new ColumnNotFound(message);
      case 4 -> new Format(message);
      case 5 -> new FileAccess(message);
      default -> new MsEntityException(status, message);
    };
  }

  /** Invalid argument or invalid data. */
  public static final class InvalidArgument extends MsEntityException {
    InvalidArgument(String message) {
      super(1, message);
    }
  }

  /** Spectrum or peak position out of range. */
  public static final class PositionOutOfRange extends MsEntityException {
    PositionOutOfRange(String message) {
      super(2, message);
    }
  }

  /** Unknown column. */
  public static final class ColumnNotFound extends MsEntityException {
    ColumnNotFound(String message) {
      super(3, message);
    }
  }

  /** Malformed or unreadable file. */
  public static final class Format extends MsEntityException {
    Format(String message) {
      super(4, message);
    }
  }

  /** File cannot be opened or written. */
  public static final class FileAccess extends MsEntityException {
    FileAccess(String message) {
      super(5, message);
    }
  }
}
