using MsEntity;
using MsEntity.Similarity;

namespace MsEntity.Similarity.Tests;

public class SimilarityTests
{
    private static readonly string Repo =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../.."));
    private static readonly string MsEntityData =
        Environment.GetEnvironmentVariable("MSENTITY_TEST_DATA_DIR") ??
        Path.GetFullPath(Path.Combine(Repo, "../msentity/tests/data"));
    private static readonly string Sample = Path.Combine(MsEntityData, "sample.msp");

    // Reference values for sample.msp, cosine, bin width 0.01.
    private static readonly float[] Row0 =
        { 1.0f, 0.7569831609725952f, 0.85833740234375f, 0.6929314732551575f, 0.6837769150733948f };

    [Fact]
    public void AllPairsAndPairScores()
    {
        using var ds = Dataset.Load(Sample);
        var matrix = SpectrumSimilarity.AllPairs(ds, ds);
        Assert.Equal(5, matrix.GetLength(0));
        for (int j = 0; j < 5; j++) Assert.Equal(Row0[j], matrix[0, j], 1e-6f);
        var scores = SpectrumSimilarity.PairScores(ds, new long[] { 0, 2 }, ds, new long[] { 1, 4 },
            new SimilarityOptions(Method: SimilarityMethod.ReverseCosine));
        Assert.Equal(0.7586880922317505f, scores[0], 1e-6f);
        Assert.Equal(0.31010058522224426f, scores[1], 1e-6f);
        Assert.Throws<SimilarityArgumentException>(() =>
            SpectrumSimilarity.AllPairs(ds, ds, new SimilarityOptions(BinWidth: 0)));
        Assert.Throws<SimilarityPositionException>(() =>
            SpectrumSimilarity.PairScores(ds, new long[] { 9 }, ds, new long[] { 0 }));
    }

    [Fact]
    public void LibrarySearch()
    {
        using var ds = Dataset.Load(Sample);
        long lastDone = -1, lastTotal = -1;
        using var result = SpectrumSimilarity.LibrarySearch(ds, ds, 0.9, querySource: "sample.msp",
            progress: (done, total) => { lastDone = done; lastTotal = total; });
        Assert.Equal((25L, 25L), (lastDone, lastTotal));
        Assert.Equal(9, result.Count);
        Assert.Equal(new[] { "index1", "index2", "cosine_similarity", "data_index1", "data_index2" },
            result.Columns);
        Assert.Equal(new object?[] { 0L, 1L, 4L, 2L, 3L, 2L, 3L, 1L, 4L }, result["index2"]);
        Assert.Contains("\"mode\":\"library_search\"", result.MetadataJson);
        Assert.True(result.HasMatchedData);
        Assert.Equal(new long[] { 0, 1, 4, 2, 3 }, result.SourceIndices(2));
        var (first, second) = result.MatchRecords(2); // pair (1, 4)
        using (first)
        using (second)
        {
            Assert.Equal(ds.GetValue(1, "Name"), first.GetValue(0, "Name"));
            Assert.Equal(ds.GetValue(4, "Name"), second.GetValue(0, "Name"));
        }
    }

    [Fact]
    public void ByKey()
    {
        using var ds = Dataset.Load(Sample);
        ds.SetSpecId("SpecID", "SP");
        using var reordered = ds.Select(4, 3, 2);
        using var result = SpectrumSimilarity.SimilarityByKey(ds, reordered);
        Assert.Equal(new[] { "SpecID", "index1", "index2", "cosine_similarity" }, result.Columns);
        Assert.Equal(new object?[] { 2L, 3L, 4L }, result["index1"]);
        Assert.Equal(new object?[] { 2L, 1L, 0L }, result["index2"]);
        Assert.False(result.HasMatchedData);
        Assert.Throws<SimilarityArgumentException>(() => result.MatchRecords(0));
        Assert.Throws<SimilarityColumnNotFoundException>(() =>
            SpectrumSimilarity.SimilarityByKey(ds, ds, "Missing"));
    }

    [Fact]
    public void FilterSortSaveLoad()
    {
        string tmp = Directory.CreateTempSubdirectory("mssim-cs-").FullName;
        try
        {
            using var ds = Dataset.Load(Sample);
            using var result = SpectrumSimilarity.LibrarySearch(ds, ds, 0.9);
            var mask = result["index1"].Select(v => (long)v! == 1).ToArray();
            using var filtered = result.Filter(mask);
            using var ranked = filtered.SortBy("cosine_similarity", ascending: false);
            Assert.Equal(new object?[] { 1L, 4L }, ranked["index2"]);
            string path = Path.Combine(tmp, "hits.mssim");
            ranked.Save(path);
            using var loaded = SimilarityDataset.Load(path);
            Assert.Equal(2, loaded.Count);
            Assert.Equal(new long[] { 1, 4 }, loaded.SourceIndices(2));
            loaded.ExportTable(Path.Combine(tmp, "hits.tsv"));
            Assert.StartsWith("index1\tindex2\tcosine_similarity", File.ReadAllText(Path.Combine(tmp, "hits.tsv")));
            Assert.Throws<SimilarityFileException>(() => SimilarityDataset.Load(Path.Combine(tmp, "missing.mssim")));
        }
        finally
        {
            Directory.Delete(tmp, recursive: true);
        }
    }

    [Fact]
    public void ReferenceFileAndStatistics()
    {
        using var result = SimilarityDataset.Load(Path.Combine(Repo, "tests/data/similarity.mssim"));
        Assert.Equal(16, result.Count);
        using (var first = result.MatchedDataset(1)) Assert.Equal(6, first.Count);
        Assert.Equal(16, result.DescribeScores().Count);
        var histogram = result.Histogram(4);
        Assert.Equal(16, histogram.Counts.Sum());
        Assert.Equal(5, histogram.Edges.Length);
        result.MetadataJson = "{\"note\": \"名前\"}";
        Assert.Equal("{\"note\": \"名前\"}", result.MetadataJson);
        Assert.Throws<SimilarityArgumentException>(() => result.MetadataJson = "[1]");
    }
}
