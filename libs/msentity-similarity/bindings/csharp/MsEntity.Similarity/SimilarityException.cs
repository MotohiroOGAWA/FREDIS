namespace MsEntity.Similarity;

/// <summary>
/// An error reported by the msentity-similarity library; the subclass tells the kind.
/// Status codes are those of msentity (MSENTITY_ERR_*).
/// </summary>
public class SimilarityException : Exception
{
    /// <summary>The C API status code (MSENTITY_ERR_*).</summary>
    public int Status { get; }

    internal SimilarityException(int status, string message) : base(message) => Status = status;

    internal static SimilarityException Create(int status, string message) => status switch
    {
        1 => new SimilarityArgumentException(message),
        2 => new SimilarityPositionException(message),
        3 => new SimilarityColumnNotFoundException(message),
        4 => new SimilarityFormatException(message),
        5 => new SimilarityFileException(message),
        _ => new SimilarityException(status, message),
    };
}

/// <summary>Invalid argument or invalid data.</summary>
public sealed class SimilarityArgumentException : SimilarityException
{
    internal SimilarityArgumentException(string message) : base(1, message) { }
}

/// <summary>Spectrum or row position out of range.</summary>
public sealed class SimilarityPositionException : SimilarityException
{
    internal SimilarityPositionException(string message) : base(2, message) { }
}

/// <summary>Unknown column.</summary>
public sealed class SimilarityColumnNotFoundException : SimilarityException
{
    internal SimilarityColumnNotFoundException(string message) : base(3, message) { }
}

/// <summary>Malformed or unreadable file or data.</summary>
public sealed class SimilarityFormatException : SimilarityException
{
    internal SimilarityFormatException(string message) : base(4, message) { }
}

/// <summary>File cannot be opened or written.</summary>
public sealed class SimilarityFileException : SimilarityException
{
    internal SimilarityFileException(string message) : base(5, message) { }
}
