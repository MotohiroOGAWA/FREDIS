using System.Runtime.InteropServices;
using static MsEntity.NativeMethods;

namespace MsEntity;

/// <summary>
/// Spectra with their metadata and peaks.
/// </summary>
/// <remarks>
/// Cell values are <c>null</c>, <see cref="bool"/>, <see cref="long"/>, <see cref="double"/>,
/// or <see cref="string"/>. <see cref="Select"/>, <see cref="Slice"/>, <see cref="Filter"/>,
/// <see cref="SortBy"/>, and <see cref="FullView"/> return views that share data with this
/// dataset; <see cref="Copy"/> makes an independent one. Dispose datasets when done.
/// </remarks>
public sealed class Dataset : IDisposable
{
    private readonly DatasetHandle _handle;

    private Dataset(DatasetHandle handle) => _handle = handle;

    /// <summary>Release the native dataset. Views created from it stay valid.</summary>
    public void Dispose() => _handle.Dispose();

    /// <summary>Version of the native library.</summary>
    public static string Version => Marshal.PtrToStringUTF8(msentity_version()) ?? "";

    private static Dataset Wrap(int status, DatasetHandle handle)
    {
        if (status != 0)
        {
            handle.Dispose();
            Check(status);
        }
        return new Dataset(handle);
    }

    // ---- creating ------------------------------------------------------------------------

    /// <summary>Load MSP, MGF, MSDS, TSV, or CSV.</summary>
    /// <param name="path">File to read.</param>
    /// <param name="fileType">"msp", "mgf", "msds", "tsv", "csv", or null to use the extension.</param>
    /// <param name="specIdPrefix">When not null, add a SpecID column (prefix + number).</param>
    /// <param name="inferTypes">Convert numeric TSV/CSV columns to numbers.</param>
    public static Dataset Load(string path, string? fileType = null, string? specIdPrefix = null,
        bool inferTypes = false)
    {
        int status = msentity_load(path, fileType, specIdPrefix, inferTypes ? 1 : 0, out var handle);
        return Wrap(status, handle);
    }

    /// <summary>Parse text in "msp", "mgf", "tsv", or "csv" format.</summary>
    public static Dataset ReadText(string text, string format, bool inferTypes = false)
    {
        int status = msentity_read_text(text, format, inferTypes ? 1 : 0, out var handle);
        return Wrap(status, handle);
    }

    /// <summary>Build a dataset without metadata; one array of (m/z, intensity) per spectrum.</summary>
    public static unsafe Dataset FromPeaks(IReadOnlyList<(double Mz, double Intensity)[]> spectra)
    {
        var data = new List<double>();
        var offsets = new long[spectra.Count + 1];
        for (int s = 0; s < spectra.Count; s++)
        {
            foreach (var (mz, intensity) in spectra[s])
            {
                data.Add(mz);
                data.Add(intensity);
            }
            offsets[s + 1] = data.Count / 2;
        }
        double[] array = data.ToArray();
        fixed (double* d = array)
        fixed (long* o = offsets)
        {
            int status = msentity_from_peaks(d, array.Length / 2, o, offsets.Length, out var handle);
            return Wrap(status, handle);
        }
    }

    /// <summary>Dataset from the bytes of an .msds file (see <see cref="ToBytes"/>).</summary>
    public static unsafe Dataset FromBytes(byte[] data)
    {
        fixed (byte* p = data.Length > 0 ? data : new byte[1])
        {
            int status = msentity_from_msds_bytes(p, data.Length, out var handle);
            return Wrap(status, handle);
        }
    }

    /// <summary>
    /// The visible data as the bytes of an .msds file. Other msentity libraries
    /// (such as msentity-similarity) exchange datasets this way.
    /// </summary>
    public byte[] ToBytes()
    {
        Check(msentity_to_msds_bytes(_handle, out IntPtr data, out long size));
        try
        {
            var bytes = new byte[size];
            Marshal.Copy(data, bytes, 0, (int)size);
            return bytes;
        }
        finally
        {
            msentity_bytes_free(data);
        }
    }

    /// <summary>Join datasets; columns are the union and missing cells are null.</summary>
    public static unsafe Dataset Concat(params Dataset[] datasets)
    {
        var pointers = new IntPtr[datasets.Length];
        bool[] added = new bool[datasets.Length];
        try
        {
            for (int i = 0; i < datasets.Length; i++)
            {
                datasets[i]._handle.DangerousAddRef(ref added[i]);
                pointers[i] = datasets[i]._handle.DangerousGetHandle();
            }
            fixed (IntPtr* p = pointers)
            {
                int status = msentity_concat(p, pointers.Length, out var handle);
                return Wrap(status, handle);
            }
        }
        finally
        {
            for (int i = 0; i < datasets.Length; i++)
            {
                if (added[i]) datasets[i]._handle.DangerousRelease();
            }
        }
    }

    // ---- shape and columns ---------------------------------------------------------------------

    /// <summary>Number of spectra.</summary>
    public long Count => msentity_size(_handle);

    /// <summary>Number of peaks of all spectra.</summary>
    public long PeakCount => msentity_n_peaks_total(_handle);

    /// <summary>Visible metadata columns in display order; assign to choose them.</summary>
    public unsafe IReadOnlyList<string> Columns
    {
        get
        {
            long n = msentity_n_columns(_handle);
            var names = new List<string>();
            for (long i = 0; i < n; i++)
            {
                Check(msentity_column_name(_handle, i, out IntPtr name));
                names.Add(TakeString(name));
            }
            return names;
        }
        set
        {
            var pointers = value.Select(Marshal.StringToCoTaskMemUTF8).ToArray();
            try
            {
                fixed (IntPtr* p = pointers) Check(msentity_set_columns(_handle, p, pointers.Length));
            }
            finally
            {
                foreach (var pointer in pointers) Marshal.FreeCoTaskMem(pointer);
            }
        }
    }

    /// <summary>Whether a column is visible.</summary>
    public bool HasColumn(string name) => msentity_has_column(_handle, name) != 0;

    // ---- values ---------------------------------------------------------------------------------------

    private static object? ReadValue(in NativeValue value) => value.Type switch
    {
        NativeValue.TypeBool => value.BoolValue != 0,
        NativeValue.TypeInt => value.IntValue,
        NativeValue.TypeFloat => value.FloatValue,
        NativeValue.TypeString => Marshal.PtrToStringUTF8(value.StringValue),
        _ => null,
    };

    /// <summary>Fill a native value; strings are allocated and recorded in <paramref name="allocated"/>.</summary>
    private static NativeValue WriteValue(object? item, List<IntPtr> allocated)
    {
        var value = new NativeValue();
        switch (item)
        {
            case null:
            case double d when double.IsNaN(d):
                value.Type = NativeValue.TypeNull;
                break;
            case bool b:
                value.Type = NativeValue.TypeBool;
                value.BoolValue = b ? 1 : 0;
                break;
            case long or int or short or sbyte or byte or ushort or uint:
                value.Type = NativeValue.TypeInt;
                value.IntValue = Convert.ToInt64(item);
                break;
            case double or float or decimal:
                value.Type = NativeValue.TypeFloat;
                value.FloatValue = Convert.ToDouble(item);
                break;
            case string s:
                value.Type = NativeValue.TypeString;
                value.StringValue = Marshal.StringToCoTaskMemUTF8(s);
                allocated.Add(value.StringValue);
                break;
            default:
                throw new ArgumentException($"unsupported value type: {item.GetType().Name}");
        }
        return value;
    }

    private static unsafe List<object?> ReadValues(long n, Func<IntPtr, long, int> fill)
    {
        var array = new NativeValue[Math.Max(1, n)];
        fixed (NativeValue* p = array)
        {
            Check(fill((IntPtr)p, n));
            try
            {
                var values = new List<object?>((int)n);
                for (long i = 0; i < n; i++) values.Add(ReadValue(array[i]));
                return values;
            }
            finally
            {
                msentity_values_clear(p, n);
            }
        }
    }

    private static unsafe void WithValues(IEnumerable<object?> items, Action<IntPtr, long> use)
    {
        var allocated = new List<IntPtr>();
        try
        {
            var array = items.Select(item => WriteValue(item, allocated)).ToArray();
            fixed (NativeValue* p = array.Length > 0 ? array : new NativeValue[1])
            {
                use((IntPtr)p, array.Length);
            }
        }
        finally
        {
            foreach (var pointer in allocated) Marshal.FreeCoTaskMem(pointer);
        }
    }

    /// <summary>One cell.</summary>
    public unsafe object? GetValue(long position, string column)
    {
        NativeValue value;
        Check(msentity_get_value(_handle, position, column, &value));
        try
        {
            return ReadValue(value);
        }
        finally
        {
            msentity_values_clear(&value, 1);
        }
    }

    /// <summary>All values of a column, one per spectrum.</summary>
    public unsafe IReadOnlyList<object?> GetColumn(string name) =>
        ReadValues(Count, (p, n) => msentity_get_column(_handle, name, (NativeValue*)p, n));

    /// <summary>Values of a column (same as <see cref="GetColumn"/>).</summary>
    public IReadOnlyList<object?> this[string column] => GetColumn(column);

    /// <summary>Set a column from one value per spectrum.</summary>
    public unsafe void SetColumn(string name, IEnumerable<object?> values) =>
        WithValues(values, (p, n) => Check(msentity_set_column(_handle, name, (NativeValue*)p, n)));

    /// <summary>Set a column to the same value for every spectrum.</summary>
    public unsafe void FillColumn(string name, object? value) =>
        WithValues(new[] { value }, (p, _) => Check(msentity_set_column_scalar(_handle, name, (NativeValue*)p)));

    /// <summary>Set one cell.</summary>
    public unsafe void SetValue(long position, string column, object? value) =>
        WithValues(new[] { value }, (p, _) => Check(msentity_set_value(_handle, position, column, (NativeValue*)p)));

    // ---- views ------------------------------------------------------------------------------------------

    /// <summary>Spectra at the given positions, in that order (negative = from the end).</summary>
    public unsafe Dataset Select(params long[] positions)
    {
        fixed (long* p = positions.Length > 0 ? positions : new long[1])
        {
            int status = msentity_select(_handle, p, positions.Length, out var view);
            return Wrap(status, view);
        }
    }

    /// <summary>Spectra <paramref name="start"/> (inclusive) to <paramref name="stop"/> (exclusive).</summary>
    public Dataset Slice(long start, long stop) => Wrap(msentity_slice(_handle, start, stop, out var view), view);

    /// <summary>Spectra whose mask entry is true (one entry per spectrum).</summary>
    public unsafe Dataset Filter(bool[] mask)
    {
        byte[] bytes = mask.Select(b => (byte)(b ? 1 : 0)).ToArray();
        fixed (byte* p = bytes.Length > 0 ? bytes : new byte[1])
        {
            int status = msentity_filter(_handle, p, bytes.Length, out var view);
            return Wrap(status, view);
        }
    }

    /// <summary>Stable sort by a column; missing values last.</summary>
    public Dataset SortBy(string column, bool ascending = true) =>
        Wrap(msentity_sort_by(_handle, column, ascending ? 1 : 0, out var view), view);

    /// <summary>An independent dataset with the visible spectra and columns.</summary>
    public Dataset Copy() => Wrap(msentity_copy(_handle, out var copy), copy);

    /// <summary>All stored spectra; with <paramref name="resetColumns"/>, also all columns.</summary>
    public Dataset FullView(bool resetColumns = false) =>
        Wrap(msentity_full_view(_handle, resetColumns ? 1 : 0, out var view), view);

    // ---- peaks ---------------------------------------------------------------------------------------------

    /// <summary>Spectrum i has peaks Offsets[i] … Offsets[i + 1] - 1.</summary>
    public unsafe long[] Offsets
    {
        get
        {
            var offsets = new long[Count + 1];
            fixed (long* p = offsets) Check(msentity_offsets(_handle, p, offsets.Length));
            return offsets;
        }
    }

    private unsafe (double[] Mz, double[] Intensity) PeakArrays()
    {
        long n = PeakCount;
        var mz = new double[n];
        var intensity = new double[n];
        fixed (double* m = mz.Length > 0 ? mz : new double[1])
        fixed (double* i = intensity.Length > 0 ? intensity : new double[1])
        {
            Check(msentity_peaks(_handle, m, i, n));
        }
        return (mz, intensity);
    }

    /// <summary>m/z of every peak of every spectrum.</summary>
    public double[] Mz => PeakArrays().Mz;

    /// <summary>Intensity of every peak of every spectrum.</summary>
    public double[] Intensity => PeakArrays().Intensity;

    /// <summary>The peaks of one spectrum.</summary>
    public unsafe Spectrum GetSpectrum(long position)
    {
        Check(msentity_spectrum_size(_handle, position, out long n));
        var mz = new double[n];
        var intensity = new double[n];
        fixed (double* m = mz.Length > 0 ? mz : new double[1])
        fixed (double* i = intensity.Length > 0 ? intensity : new double[1])
        {
            Check(msentity_spectrum_peaks(_handle, position, m, i, n));
        }
        return new Spectrum(mz, intensity);
    }

    /// <summary>In place: the highest peak of each spectrum becomes <paramref name="scale"/>.</summary>
    public void Normalize(double scale = 1.0) => Check(msentity_normalize(_handle, scale));

    /// <summary>In place: sort the peaks of each spectrum by m/z.</summary>
    public void SortPeaksByMz(bool ascending = true) => Check(msentity_sort_peaks(_handle, 0, ascending ? 1 : 0));

    /// <summary>In place: sort the peaks of each spectrum by intensity.</summary>
    public void SortPeaksByIntensity(bool ascending = false) =>
        Check(msentity_sort_peaks(_handle, 1, ascending ? 1 : 0));

    // ---- peak metadata ---------------------------------------------------------------------------------------

    /// <summary>Visible peak metadata columns.</summary>
    public IReadOnlyList<string> PeakColumns
    {
        get
        {
            long n = msentity_n_peak_columns(_handle);
            var names = new List<string>();
            for (long i = 0; i < n; i++)
            {
                Check(msentity_peak_column_name(_handle, i, out IntPtr name));
                names.Add(TakeString(name));
            }
            return names;
        }
    }

    /// <summary>Values of a peak column, one per peak.</summary>
    public unsafe IReadOnlyList<object?> GetPeakColumn(string name) =>
        ReadValues(PeakCount, (p, n) => msentity_get_peak_column(_handle, name, (NativeValue*)p, n));

    /// <summary>Set a peak column from one value per peak.</summary>
    public unsafe void SetPeakColumn(string name, IEnumerable<object?> values) =>
        WithValues(values, (p, n) => Check(msentity_set_peak_column(_handle, name, (NativeValue*)p, n)));

    // ---- dataset-level metadata ---------------------------------------------------------------------------------

    /// <summary>Free-text description.</summary>
    public string Description
    {
        get
        {
            Check(msentity_get_description(_handle, out IntPtr text));
            return TakeString(text);
        }
        set => Check(msentity_set_description(_handle, value));
    }

    /// <summary>A copy of the attributes in insertion order.</summary>
    public IReadOnlyList<KeyValuePair<string, string>> Attributes
    {
        get
        {
            long n = msentity_n_attributes(_handle);
            var result = new List<KeyValuePair<string, string>>();
            for (long i = 0; i < n; i++)
            {
                Check(msentity_attribute(_handle, i, out IntPtr key, out IntPtr value));
                result.Add(new(TakeString(key), TakeString(value)));
            }
            return result;
        }
    }

    /// <summary>Add or replace an attribute.</summary>
    public void SetAttribute(string key, string value) => Check(msentity_set_attribute(_handle, key, value));

    /// <summary>Returns true if the key existed.</summary>
    public bool RemoveAttribute(string key)
    {
        Check(msentity_remove_attribute(_handle, key, out int removed));
        return removed != 0;
    }

    /// <summary>Tags in insertion order.</summary>
    public IReadOnlyList<string> Tags
    {
        get
        {
            long n = msentity_n_tags(_handle);
            var result = new List<string>();
            for (long i = 0; i < n; i++)
            {
                Check(msentity_tag(_handle, i, out IntPtr tag));
                result.Add(TakeString(tag));
            }
            return result;
        }
    }

    /// <summary>Returns false if the tag was already present.</summary>
    public bool AddTag(string tag)
    {
        Check(msentity_add_tag(_handle, tag, out int added));
        return added != 0;
    }

    /// <summary>Returns true if the tag existed.</summary>
    public bool RemoveTag(string tag)
    {
        Check(msentity_remove_tag(_handle, tag, out int removed));
        return removed != 0;
    }

    /// <summary>Sequential zero-padded IDs (SP01 … SP12); false if the column exists and not overwrite.</summary>
    public bool SetSpecId(string column = "SpecID", string prefix = "", bool overwrite = false, long start = 1)
    {
        Check(msentity_set_spec_id(_handle, column, prefix, overwrite ? 1 : 0, start, out int assigned));
        return assigned != 0;
    }

    // ---- saving ---------------------------------------------------------------------------------------------------

    /// <summary>Save as MSDS, MSP, MGF, TSV, or CSV.</summary>
    /// <param name="path">Target file.</param>
    /// <param name="format">"msds", "msp", "mgf", "tsv", "csv", or null to use the extension.</param>
    /// <param name="saveView">MSDS: store only the visible data (true) or everything with the view.</param>
    /// <param name="append">MSDS: add a dataset group instead of replacing the file.</param>
    public void Save(string path, string? format = null, bool saveView = true, bool append = false) =>
        Check(msentity_save(_handle, path, format, saveView ? 1 : 0, append ? 1 : 0));

    /// <inheritdoc/>
    public override string ToString() =>
        $"Dataset(n_spectra={Count}, n_peaks={PeakCount}, columns=[{string.Join(", ", Columns)}])";
}

/// <summary>The peaks of one spectrum.</summary>
/// <param name="Mz">m/z values.</param>
/// <param name="Intensity">Intensities.</param>
public sealed record Spectrum(double[] Mz, double[] Intensity)
{
    /// <summary>Number of peaks.</summary>
    public int Count => Mz.Length;
}
