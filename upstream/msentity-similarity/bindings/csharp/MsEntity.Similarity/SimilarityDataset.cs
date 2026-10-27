using System.Runtime.InteropServices;
using static MsEntity.Similarity.NativeMethods;

namespace MsEntity.Similarity;

/// <summary>Summary of the score column; NaN statistics when empty.</summary>
public sealed record ScoreSummary(long Count, double Mean, double Q1, double Median, double Q3, double Min,
    double Max);

/// <summary>Score counts over [0, 1]; <see cref="Edges"/> has one more entry than <see cref="Counts"/>.</summary>
public sealed record ScoreHistogram(long[] Counts, double[] Edges);

/// <summary>
/// A similarity result (one row per spectrum pair), as stored in .mssim files.
/// </summary>
/// <remarks>
/// Columns: key columns (by-key results), <c>index1</c>, <c>index2</c> (positions in the input
/// datasets), <c>cosine_similarity</c> (the score of any method), and <c>data_index1</c> /
/// <c>data_index2</c> when the compared spectra are embedded. Cell values are <c>null</c>,
/// <see cref="bool"/>, <see cref="long"/>, <see cref="double"/>, or <see cref="string"/>.
/// Dispose results when done.
/// </remarks>
public sealed class SimilarityDataset : IDisposable
{
    private readonly ResultHandle _handle;

    private SimilarityDataset(ResultHandle handle) => _handle = handle;

    /// <summary>Release the native result.</summary>
    public void Dispose() => _handle.Dispose();

    internal static SimilarityDataset Wrap(int status, ResultHandle handle)
    {
        if (status != 0)
        {
            handle.Dispose();
            Check(status);
        }
        return new SimilarityDataset(handle);
    }

    /// <summary>Read an .mssim file.</summary>
    public static SimilarityDataset Load(string path) => Wrap(mssim_load(path, out var handle), handle);

    /// <summary>Write an .mssim file (embedded spectra are reduced to those still referenced).</summary>
    public void Save(string path) => Check(mssim_save(_handle, path));

    /// <summary>Write the table only, as .parquet, .csv, or .tsv.</summary>
    public void ExportTable(string path) => Check(mssim_export_table(_handle, path));

    // ---- table ------------------------------------------------------------------------------

    /// <summary>Number of rows (spectrum pairs).</summary>
    public long Count => mssim_size(_handle);

    /// <summary>Column names.</summary>
    public IReadOnlyList<string> Columns
    {
        get
        {
            long n = mssim_n_columns(_handle);
            var names = new List<string>((int)n);
            for (long i = 0; i < n; i++)
            {
                Check(mssim_column_name(_handle, i, out IntPtr name));
                names.Add(TakeString(name));
            }
            return names;
        }
    }

    private static object? ReadValue(in NativeValue value) => value.Type switch
    {
        NativeValue.TypeBool => value.BoolValue != 0,
        NativeValue.TypeInt => value.IntValue,
        NativeValue.TypeFloat => value.FloatValue,
        NativeValue.TypeString => Marshal.PtrToStringUTF8(value.StringValue),
        _ => null,
    };

    /// <summary>All values of a column.</summary>
    public unsafe IReadOnlyList<object?> GetColumn(string name)
    {
        long n = Count;
        var array = new NativeValue[Math.Max(1, n)];
        fixed (NativeValue* p = array)
        {
            Check(mssim_get_column(_handle, name, p, n));
            try
            {
                var values = new List<object?>((int)n);
                for (long i = 0; i < n; i++) values.Add(ReadValue(array[i]));
                return values;
            }
            finally
            {
                mssim_values_clear(p, n);
            }
        }
    }

    /// <summary>Values of a column (same as <see cref="GetColumn"/>).</summary>
    public IReadOnlyList<object?> this[string column] => GetColumn(column);

    // ---- metadata ---------------------------------------------------------------------------

    /// <summary>Calculation metadata (parameters, sources, created_at, …) as a JSON object.</summary>
    public string MetadataJson
    {
        get
        {
            Check(mssim_get_metadata_json(_handle, out IntPtr json));
            return TakeString(json);
        }
        set => Check(mssim_set_metadata_json(_handle, value));
    }

    // ---- matched data -----------------------------------------------------------------------

    /// <summary>True when the compared spectra are embedded.</summary>
    public bool HasMatchedData => mssim_has_matched_data(_handle) != 0;

    /// <summary>
    /// The embedded spectra of one side (1 = query / first, 2 = reference / second); only those
    /// the rows refer to.
    /// </summary>
    public Dataset MatchedDataset(int side)
    {
        Check(mssim_matched_dataset(_handle, side, out IntPtr data, out long size));
        try
        {
            var bytes = new byte[size];
            Marshal.Copy(data, bytes, 0, (int)size);
            return Dataset.FromBytes(bytes);
        }
        finally
        {
            mssim_bytes_free(data);
        }
    }

    /// <summary>For each embedded spectrum of <paramref name="side"/>, its position in the original input.</summary>
    public unsafe long[] SourceIndices(int side)
    {
        var output = new long[mssim_matched_size(_handle, side)];
        fixed (long* p = output.Length > 0 ? output : new long[1])
        {
            Check(mssim_source_indices(_handle, side, p, output.Length));
        }
        return output;
    }

    /// <summary>The two spectra compared in <paramref name="row"/>, as one-spectrum datasets.</summary>
    public (Dataset First, Dataset Second) MatchRecords(long row)
    {
        Check(mssim_match_positions(_handle, row, out long d1, out long d2));
        using var first = MatchedDataset(1);
        using var second = MatchedDataset(2);
        return (first.Select(d1), second.Select(d2));
    }

    // ---- selection --------------------------------------------------------------------------

    /// <summary>Rows where <paramref name="mask"/> is true.</summary>
    public unsafe SimilarityDataset Filter(bool[] mask)
    {
        var bytes = mask.Select(m => (byte)(m ? 1 : 0)).ToArray();
        fixed (byte* p = bytes.Length > 0 ? bytes : new byte[1])
        {
            return Wrap(mssim_filter(_handle, p, bytes.Length, out var output), output);
        }
    }

    /// <summary>Stable sort by several columns; missing values last.</summary>
    public unsafe SimilarityDataset SortBy(IReadOnlyList<string> columns, IReadOnlyList<bool> ascending)
    {
        if (columns.Count != ascending.Count)
            throw new SimilarityArgumentException("ascending must have one entry per column");
        var names = columns.Select(c => Marshal.StringToCoTaskMemUTF8(c)).ToArray();
        try
        {
            var orders = ascending.Select(a => a ? 1 : 0).ToArray();
            fixed (IntPtr* pn = names.Length > 0 ? names : new IntPtr[1])
            fixed (int* po = orders.Length > 0 ? orders : new int[1])
            {
                return Wrap(mssim_sort_by(_handle, pn, po, names.Length, out var output), output);
            }
        }
        finally
        {
            foreach (var name in names) Marshal.FreeCoTaskMem(name);
        }
    }

    /// <summary>Stable sort by one column; missing values last.</summary>
    public SimilarityDataset SortBy(string column, bool ascending = true) =>
        SortBy(new[] { column }, new[] { ascending });

    // ---- statistics -------------------------------------------------------------------------

    /// <summary>count, mean, quartiles, min, and max of the score column.</summary>
    public unsafe ScoreSummary DescribeScores()
    {
        double* v = stackalloc double[7];
        Check(mssim_describe_scores(_handle, v));
        return new ScoreSummary((long)v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }

    /// <summary>Score histogram over [0, 1] with 1–200 bins.</summary>
    public unsafe ScoreHistogram Histogram(int bins = 20)
    {
        if (bins < 1 || bins > 200) throw new SimilarityArgumentException("bins must be between 1 and 200");
        var counts = new long[bins];
        var edges = new double[bins + 1];
        fixed (long* pc = counts)
        fixed (double* pe = edges)
        {
            Check(mssim_histogram(_handle, bins, pc, pe));
        }
        return new ScoreHistogram(counts, edges);
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"SimilarityDataset(rows={Count}, columns=[{string.Join(", ", Columns)}], " +
        $"matched_data={(HasMatchedData ? "embedded" : "lightweight")})";
}
