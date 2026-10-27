using static MsEntity.Similarity.NativeMethods;

namespace MsEntity.Similarity;

/// <summary>The scoring method.</summary>
public enum SimilarityMethod
{
    /// <summary>Binned cosine similarity.</summary>
    Cosine = 0,
    /// <summary>Cosine where the query norm counts only peaks matched in the reference.</summary>
    ReverseCosine = 1,
}

/// <summary>Score settings.</summary>
/// <param name="Method">Cosine or reverse cosine.</param>
/// <param name="BinWidth">m/z bin width (&gt; 0); peaks are binned by floor(mz / BinWidth).</param>
/// <param name="IntensityExponent">Intensities are raised to this power (&gt; 0) before scoring.</param>
/// <param name="Threads">Worker threads; 0 = all hardware threads.</param>
public sealed record SimilarityOptions(
    SimilarityMethod Method = SimilarityMethod.Cosine,
    double BinWidth = 0.01,
    double IntensityExponent = 1.0,
    int Threads = 0)
{
    internal NativeOptions ToNative() => new()
    {
        Method = (int)Method,
        Threads = Threads,
        BinWidth = BinWidth,
        IntensityExponent = IntensityExponent,
    };
}

/// <summary>
/// Similarity calculations on msentity datasets.
/// </summary>
/// <example>
/// <code>
/// using var query = Dataset.Load("query.msp");
/// using var library = Dataset.Load("library.msp");
/// using var hits = SpectrumSimilarity.LibrarySearch(query, library, threshold: 0.8);
/// hits["cosine_similarity"];
/// hits.Save("hits.mssim");
/// </code>
/// </example>
/// <remarks>Datasets are handed to the C library as .msds bytes (<see cref="Dataset.ToBytes"/>).</remarks>
public static class SpectrumSimilarity
{
    /// <summary>Version of the native library.</summary>
    public static string Version => System.Runtime.InteropServices.Marshal.PtrToStringUTF8(mssim_version()) ?? "";

    private static byte[] Bytes(Dataset dataset) => dataset.ToBytes();

    /// <summary>Scores of ds1[index1[k]] vs ds2[index2[k]].</summary>
    public static unsafe float[] PairScores(Dataset ds1, long[] index1, Dataset ds2, long[] index2,
        SimilarityOptions? options = null)
    {
        if (index1.Length != index2.Length)
            throw new SimilarityArgumentException("index1 and index2 must have the same length");
        byte[] a = Bytes(ds1), b = Bytes(ds2);
        var output = new float[index1.Length];
        var native = (options ?? new SimilarityOptions()).ToNative();
        fixed (byte* pa = a, pb = b)
        fixed (long* i1 = index1.Length > 0 ? index1 : new long[1], i2 = index2.Length > 0 ? index2 : new long[1])
        fixed (float* po = output.Length > 0 ? output : new float[1])
        {
            Check(mssim_pair_scores(pa, a.Length, i1, pb, b.Length, i2, index1.Length, native, po));
        }
        return output;
    }

    /// <summary>ds1.Count × ds2.Count score matrix.</summary>
    public static unsafe float[,] AllPairs(Dataset ds1, Dataset ds2, SimilarityOptions? options = null)
    {
        byte[] a = Bytes(ds1), b = Bytes(ds2);
        var output = new float[ds1.Count, ds2.Count];
        var native = (options ?? new SimilarityOptions()).ToNative();
        fixed (byte* pa = a, pb = b)
        fixed (float* po = output.Length > 0 ? output : new float[1, 1])
        {
            Check(mssim_all_pairs(pa, a.Length, pb, b.Length, native, po, output.LongLength));
        }
        return output;
    }

    /// <summary>All (query, reference) pairs scoring at least <paramref name="threshold"/>.</summary>
    /// <param name="query">Query spectra.</param>
    /// <param name="reference">Reference (library) spectra.</param>
    /// <param name="threshold">Minimum score in [0, 1].</param>
    /// <param name="options">Score settings.</param>
    /// <param name="includeMatchedData">Embed the compared spectra in the result.</param>
    /// <param name="querySource">Recorded in the metadata (for example a file name).</param>
    /// <param name="referenceSource">Recorded in the metadata.</param>
    /// <param name="progress">Receives (completed pairs, total pairs); called from worker threads.</param>
    public static unsafe SimilarityDataset LibrarySearch(Dataset query, Dataset reference, double threshold = 0.8,
        SimilarityOptions? options = null, bool includeMatchedData = true, string? querySource = null,
        string? referenceSource = null, Action<long, long>? progress = null)
    {
        byte[] a = Bytes(query), b = Bytes(reference);
        var native = (options ?? new SimilarityOptions()).ToNative();
        NativeProgress? callback = progress is null ? null : (done, total, _) =>
        {
            // An exception must not unwind into native code.
            try { progress(done, total); }
            catch (Exception e) { Console.Error.WriteLine(e); }
        };
        int status;
        ResultHandle handle;
        fixed (byte* pa = a, pb = b)
        {
            status = mssim_library_search(pa, a.Length, pb, b.Length, threshold, native,
                includeMatchedData ? 1 : 0, querySource, referenceSource, callback, IntPtr.Zero, out handle);
        }
        GC.KeepAlive(callback);
        return SimilarityDataset.Wrap(status, handle);
    }

    /// <summary>
    /// Scores of the spectrum pairs whose <paramref name="key1"/> / <paramref name="key2"/> values
    /// are equal (each value must be unique within its dataset).
    /// </summary>
    public static unsafe SimilarityDataset SimilarityByKey(Dataset ds1, Dataset ds2, string key1 = "SpecID",
        string key2 = "SpecID", SimilarityOptions? options = null, bool includeMatchedData = false,
        string? source1 = null, string? source2 = null)
    {
        byte[] a = Bytes(ds1), b = Bytes(ds2);
        var native = (options ?? new SimilarityOptions()).ToNative();
        fixed (byte* pa = a, pb = b)
        {
            int status = mssim_similarity_by_key(pa, a.Length, pb, b.Length, key1, key2, native,
                includeMatchedData ? 1 : 0, source1, source2, out var handle);
            return SimilarityDataset.Wrap(status, handle);
        }
    }
}
