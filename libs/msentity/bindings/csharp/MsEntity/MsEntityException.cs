namespace MsEntity;

/// <summary>An error reported by the msentity library; the subclass tells the kind.</summary>
public class MsEntityException : Exception
{
    /// <summary>The C API status code (MSENTITY_ERR_*).</summary>
    public int Status { get; }

    internal MsEntityException(int status, string message) : base(message) => Status = status;

    internal static MsEntityException Create(int status, string message) => status switch
    {
        1 => new MsEntityArgumentException(message),
        2 => new MsEntityPositionException(message),
        3 => new MsEntityColumnNotFoundException(message),
        4 => new MsEntityFormatException(message),
        5 => new MsEntityFileException(message),
        _ => new MsEntityException(status, message),
    };
}

/// <summary>Invalid argument or invalid data.</summary>
public sealed class MsEntityArgumentException : MsEntityException
{
    internal MsEntityArgumentException(string message) : base(1, message) { }
}

/// <summary>Spectrum or peak position out of range.</summary>
public sealed class MsEntityPositionException : MsEntityException
{
    internal MsEntityPositionException(string message) : base(2, message) { }
}

/// <summary>Unknown column.</summary>
public sealed class MsEntityColumnNotFoundException : MsEntityException
{
    internal MsEntityColumnNotFoundException(string message) : base(3, message) { }
}

/// <summary>Malformed or unreadable file.</summary>
public sealed class MsEntityFormatException : MsEntityException
{
    internal MsEntityFormatException(string message) : base(4, message) { }
}

/// <summary>File cannot be opened or written.</summary>
public sealed class MsEntityFileException : MsEntityException
{
    internal MsEntityFileException(string message) : base(5, message) { }
}
