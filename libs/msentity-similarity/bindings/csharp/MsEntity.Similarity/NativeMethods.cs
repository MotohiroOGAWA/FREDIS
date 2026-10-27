using System.Reflection;
using System.Runtime.InteropServices;

namespace MsEntity.Similarity;

/// <summary>struct msentity_value (msentity/c/msentity_c.h).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeValue
{
    public const int TypeNull = 0, TypeBool = 1, TypeInt = 2, TypeFloat = 3, TypeString = 4;

    public int Type;
    public int BoolValue;
    public long IntValue;
    public double FloatValue;
    public IntPtr StringValue;
}

/// <summary>struct mssim_options.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeOptions
{
    public int Method;
    public int Threads;
    public double BinWidth;
    public double IntensityExponent;
}

/// <summary>void (*mssim_progress_fn)(int64_t done, int64_t total, void* user_data)</summary>
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate void NativeProgress(long done, long total, IntPtr userData);

/// <summary>Owns a mssim_result* and frees it with mssim_free.</summary>
internal sealed class ResultHandle : SafeHandle
{
    public ResultHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.mssim_free(handle);
        return true;
    }
}

/// <summary>P/Invoke declarations of the msentity-similarity C API.</summary>
internal static class NativeMethods
{
    private const string Library = "msentity_similarity_c";

    static NativeMethods()
    {
        // MSSIM_C_LIBRARY overrides the normal search (application directory,
        // runtimes/<rid>/native in a NuGet package, system library path).
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, Resolve);
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Library) return IntPtr.Zero;
        string? explicitPath = Environment.GetEnvironmentVariable("MSSIM_C_LIBRARY");
        if (!string.IsNullOrEmpty(explicitPath)) return NativeLibrary.Load(explicitPath);
        return NativeLibrary.TryLoad(name, assembly, searchPath, out IntPtr loaded) ? loaded : IntPtr.Zero;
    }

    private const CallingConvention Cdecl = CallingConvention.Cdecl;
    private const UnmanagedType Utf8 = UnmanagedType.LPUTF8Str;

    [DllImport(Library, CallingConvention = Cdecl)] public static extern IntPtr mssim_version();
    [DllImport(Library, CallingConvention = Cdecl)] public static extern IntPtr mssim_last_error();
    [DllImport(Library, CallingConvention = Cdecl)] public static extern void mssim_string_free(IntPtr text);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern void mssim_bytes_free(IntPtr data);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern unsafe void mssim_values_clear(NativeValue* values, long n);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_pair_scores(byte* ds1, long ds1Size, long* index1, byte* ds2, long ds2Size,
        long* index2, long n, in NativeOptions options, float* output);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_all_pairs(byte* ds1, long ds1Size, byte* ds2, long ds2Size,
        in NativeOptions options, float* output, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_library_search(byte* query, long querySize, byte* reference,
        long referenceSize, double threshold, in NativeOptions options, int includeMatchedData,
        [MarshalAs(Utf8)] string? querySource, [MarshalAs(Utf8)] string? referenceSource,
        NativeProgress? progress, IntPtr userData, out ResultHandle result);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_similarity_by_key(byte* ds1, long ds1Size, byte* ds2, long ds2Size,
        [MarshalAs(Utf8)] string key1, [MarshalAs(Utf8)] string key2, in NativeOptions options,
        int includeMatchedData, [MarshalAs(Utf8)] string? source1, [MarshalAs(Utf8)] string? source2,
        out ResultHandle result);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_load([MarshalAs(Utf8)] string path, out ResultHandle result);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_save(ResultHandle result, [MarshalAs(Utf8)] string path);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern void mssim_free(IntPtr result);

    [DllImport(Library, CallingConvention = Cdecl)] public static extern long mssim_size(ResultHandle result);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern long mssim_n_columns(ResultHandle result);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_column_name(ResultHandle result, long index, out IntPtr name);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_get_column(ResultHandle result, [MarshalAs(Utf8)] string column,
        NativeValue* output, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_get_metadata_json(ResultHandle result, out IntPtr json);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_set_metadata_json(ResultHandle result, [MarshalAs(Utf8)] string json);

    [DllImport(Library, CallingConvention = Cdecl)] public static extern int mssim_has_matched_data(ResultHandle result);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_matched_dataset(ResultHandle result, int side, out IntPtr data, out long size);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern long mssim_matched_size(ResultHandle result, int side);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_source_indices(ResultHandle result, int side, long* output, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_match_positions(ResultHandle result, long row, out long dataIndex1,
        out long dataIndex2);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_filter(ResultHandle result, byte* mask, long n, out ResultHandle output);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_sort_by(ResultHandle result, IntPtr* columns, int* ascending, long n,
        out ResultHandle output);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_describe_scores(ResultHandle result, double* output);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int mssim_histogram(ResultHandle result, int bins, long* counts, double* edges);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int mssim_export_table(ResultHandle result, [MarshalAs(Utf8)] string path);

    /// <summary>Throw the exception matching a non-zero status.</summary>
    public static void Check(int status)
    {
        if (status == 0) return;
        string message = Marshal.PtrToStringUTF8(mssim_last_error()) ?? "";
        throw SimilarityException.Create(status, message);
    }

    /// <summary>Read and free a string returned by the library.</summary>
    public static string TakeString(IntPtr pointer)
    {
        try
        {
            return Marshal.PtrToStringUTF8(pointer) ?? "";
        }
        finally
        {
            mssim_string_free(pointer);
        }
    }
}
