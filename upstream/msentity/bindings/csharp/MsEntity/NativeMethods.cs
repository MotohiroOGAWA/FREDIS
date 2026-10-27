using System.Reflection;
using System.Runtime.InteropServices;

namespace MsEntity;

/// <summary>struct msentity_value (include/msentity/c/msentity_c.h).</summary>
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

/// <summary>Owns a msentity_dataset* and frees it with msentity_free.</summary>
internal sealed class DatasetHandle : SafeHandle
{
    public DatasetHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.msentity_free(handle);
        return true;
    }
}

/// <summary>P/Invoke declarations of the msentity C API.</summary>
internal static class NativeMethods
{
    private const string Library = "msentity_c";

    static NativeMethods()
    {
        // MSENTITY_C_LIBRARY overrides the normal search (application directory,
        // runtimes/<rid>/native in a NuGet package, system library path).
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, Resolve);
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Library) return IntPtr.Zero;
        string? explicitPath = Environment.GetEnvironmentVariable("MSENTITY_C_LIBRARY");
        if (!string.IsNullOrEmpty(explicitPath)) return NativeLibrary.Load(explicitPath);
        return NativeLibrary.TryLoad(name, assembly, searchPath, out IntPtr loaded) ? loaded : IntPtr.Zero;
    }

    private const CallingConvention Cdecl = CallingConvention.Cdecl;

    [DllImport(Library, CallingConvention = Cdecl)] public static extern IntPtr msentity_version();
    [DllImport(Library, CallingConvention = Cdecl)] public static extern IntPtr msentity_last_error();
    [DllImport(Library, CallingConvention = Cdecl)] public static extern void msentity_string_free(IntPtr text);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern unsafe void msentity_values_clear(NativeValue* values, long n);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_load([MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? fileType,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? specIdPrefix, int inferTypes, out DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_read_text([MarshalAs(UnmanagedType.LPUTF8Str)] string text,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string format, int inferTypes, out DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_from_peaks(double* data, long nPeaks, long* offsets, long nOffsets,
        out DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_concat(IntPtr* datasets, long n, out DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern void msentity_free(IntPtr dataset);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_save(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? format, int saveView, int append);

    [DllImport(Library, CallingConvention = Cdecl)] public static extern long msentity_size(DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern long msentity_n_peaks_total(DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern long msentity_n_columns(DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern int msentity_column_name(DatasetHandle dataset, long index, out IntPtr name);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_has_column(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern unsafe int msentity_set_columns(DatasetHandle dataset, IntPtr* names, long n);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_get_value(DatasetHandle dataset, long position,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* value);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_get_column(DatasetHandle dataset,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* values, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_set_column(DatasetHandle dataset,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* values, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_set_column_scalar(DatasetHandle dataset,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* value);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_set_value(DatasetHandle dataset, long position,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* value);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_select(DatasetHandle dataset, long* positions, long n, out DatasetHandle view);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_slice(DatasetHandle dataset, long start, long stop, out DatasetHandle view);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_filter(DatasetHandle dataset, byte* mask, long n, out DatasetHandle view);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_sort_by(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string column,
        int ascending, out DatasetHandle view);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern int msentity_copy(DatasetHandle dataset, out DatasetHandle copy);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_full_view(DatasetHandle dataset, int resetColumns, out DatasetHandle view);

    [DllImport(Library, CallingConvention = Cdecl)] public static extern unsafe int msentity_offsets(DatasetHandle dataset, long* offsets, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_peaks(DatasetHandle dataset, double* mz, double* intensity, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_spectrum_size(DatasetHandle dataset, long position, out long size);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_spectrum_peaks(DatasetHandle dataset, long position, double* mz,
        double* intensity, long n);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern int msentity_normalize(DatasetHandle dataset, double scale);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern int msentity_sort_peaks(DatasetHandle dataset, int by, int ascending);

    [DllImport(Library, CallingConvention = Cdecl)] public static extern long msentity_n_peak_columns(DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_peak_column_name(DatasetHandle dataset, long index, out IntPtr name);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_get_peak_column(DatasetHandle dataset,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* values, long n);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_set_peak_column(DatasetHandle dataset,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string column, NativeValue* values, long n);

    [DllImport(Library, CallingConvention = Cdecl)] public static extern int msentity_get_description(DatasetHandle dataset, out IntPtr text);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_set_description(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string text);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern long msentity_n_attributes(DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_attribute(DatasetHandle dataset, long index, out IntPtr key, out IntPtr value);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_set_attribute(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string key,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string value);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_remove_attribute(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string key,
        out int removed);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern long msentity_n_tags(DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern int msentity_tag(DatasetHandle dataset, long index, out IntPtr tag);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_add_tag(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string tag, out int added);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_remove_tag(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string tag, out int removed);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_set_spec_id(DatasetHandle dataset, [MarshalAs(UnmanagedType.LPUTF8Str)] string column,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string prefix, int overwrite, long start, out int assigned);

    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern int msentity_to_msds_bytes(DatasetHandle dataset, out IntPtr data, out long size);
    [DllImport(Library, CallingConvention = Cdecl)]
    public static extern unsafe int msentity_from_msds_bytes(byte* data, long size, out DatasetHandle dataset);
    [DllImport(Library, CallingConvention = Cdecl)] public static extern void msentity_bytes_free(IntPtr data);

    /// <summary>Throw the exception matching a non-zero status.</summary>
    public static void Check(int status)
    {
        if (status == 0) return;
        string message = Marshal.PtrToStringUTF8(msentity_last_error()) ?? "";
        throw MsEntityException.Create(status, message);
    }

    /// <summary>Read and free a string returned through a char** parameter.</summary>
    public static string TakeString(IntPtr pointer)
    {
        try
        {
            return Marshal.PtrToStringUTF8(pointer) ?? "";
        }
        finally
        {
            msentity_string_free(pointer);
        }
    }
}
