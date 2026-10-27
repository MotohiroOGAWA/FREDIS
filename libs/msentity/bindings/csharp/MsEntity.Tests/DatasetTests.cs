using MsEntity;

namespace MsEntity.Tests;

public class DatasetTests
{
    private static readonly string Repo =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../.."));
    private static readonly string Example = Path.Combine(Repo, "docs/examples/example.msp");
    private static readonly string Data = Path.Combine(Repo, "tests/data");

    [Fact]
    public void LoadsTheExample()
    {
        using var ds = Dataset.Load(Example);
        Assert.Equal(2, ds.Count);
        Assert.Equal(7, ds.PeakCount);
        Assert.Equal(new[] { "Name", "PrecursorMZ", "AdductType", "Formula", "CollisionEnergy", "NumPeaks" },
            ds.Columns);
        Assert.Equal(new object?[] { "MassSpecID0000001", "MassSpecID0000002" }, ds["Name"]);
        Assert.Equal("20.0", ds.GetValue(1, "CollisionEnergy"));
        Assert.True(ds.HasColumn("NumPeaks"));
    }

    [Fact]
    public void ReadsAllFormats()
    {
        Assert.Equal(5, Dataset.Load(Path.Combine(Data, "sample.msp")).Count);
        Assert.Equal(6, Dataset.Load(Path.Combine(Data, "sample.mgf")).Count);
        Assert.Equal(6, Dataset.Load(Path.Combine(Data, "sample.msds")).Count);
        Assert.Equal("SP1", Dataset.Load(Path.Combine(Data, "sample.msp"), specIdPrefix: "SP").GetValue(0, "SpecID"));

        const string tsv = "Name\tID\tPeak\nA\t007\t1,2\n";
        Assert.Equal("007", Dataset.ReadText(tsv, "tsv").GetValue(0, "ID"));
        Assert.Equal(7L, Dataset.ReadText(tsv, "tsv", inferTypes: true).GetValue(0, "ID"));
    }

    [Fact]
    public void ErrorsBecomeExceptions()
    {
        Assert.Throws<MsEntityFileException>(() => Dataset.Load("/nonexistent/x.msds"));
        using var ds = Dataset.Load(Example);
        Assert.Throws<MsEntityColumnNotFoundException>(() => ds.GetColumn("Missing"));
        Assert.Throws<MsEntityPositionException>(() => ds.GetValue(5, "Name"));
        Assert.Throws<MsEntityArgumentException>(() => ds.SetColumn("Bad", new object?[] { 1, 2, 3 }));
        Assert.Throws<MsEntityArgumentException>(() => Dataset.ReadText("x", "xml"));
    }

    [Fact]
    public void PeaksAndPeakMetadata()
    {
        using var ds = Dataset.Load(Example);
        Assert.Equal(new long[] { 0, 4, 7 }, ds.Offsets);
        Assert.Equal(7, ds.Mz.Length);
        var first = ds.GetSpectrum(0);
        Assert.Equal(new[] { 91.0542, 125.0233, 185.0961, 246.1125 }, first.Mz);
        Assert.Equal(new[] { 0.245, 1.0, 0.349, 0.735 }, first.Intensity);
        Assert.Equal(new[] { "formula" }, ds.PeakColumns);
        Assert.Equal(new object?[] { "C7H7+", "C6H5O3+", "", "", "C7H7+", "", "" }, ds.GetPeakColumn("formula"));

        ds.Normalize(100.0);
        ds.SortPeaksByIntensity();
        var sorted = ds.GetSpectrum(0).Intensity;
        Assert.Equal(new[] { 100.0, 73.5, 34.9, 24.5 }, sorted.Select(v => Math.Round(v, 9)));
        ds.SortPeaksByMz();
        Assert.Equal(91.0542, ds.GetSpectrum(0).Mz[0]);
    }

    [Fact]
    public void ViewsShareDataAndCopiesDoNot()
    {
        var ds = Dataset.Load(Example);
        Assert.Equal(new object?[] { "MassSpecID0000002" }, ds.Slice(1, 2)["Name"]);
        Assert.Equal(new object?[] { "MassSpecID0000002", "MassSpecID0000001" }, ds.Select(1, 0)["Name"]);
        Assert.Equal(new object?[] { "MassSpecID0000002" }, ds.Filter(new[] { false, true })["Name"]);
        Assert.Equal("MassSpecID0000002", ds.SortBy("CollisionEnergy").GetValue(0, "Name"));

        var view = ds.Select(1);
        view.FillColumn("Checked", true);
        Assert.False(ds.HasColumn("Checked"));
        Assert.Equal(new object?[] { null, true }, ds.FullView(resetColumns: true)["Checked"]);
        var copy = view.Copy();
        copy.FillColumn("Checked", false);
        Assert.Equal(new object?[] { null, true }, ds.FullView(resetColumns: true)["Checked"]);
        Assert.Equal(new long[] { 0, 3 }, copy.Offsets);

        var survivor = ds.Select(0);
        ds.Dispose();
        Assert.Equal(new object?[] { "MassSpecID0000001" }, survivor["Name"]);
    }

    [Fact]
    public void BuildEditAndSave()
    {
        string tmp = Directory.CreateTempSubdirectory("msentity-cs-").FullName;
        try
        {
            using var ds = Dataset.FromPeaks(new[]
            {
                new[] { (100.0, 1.0), (150.0, 0.5) },
                new[] { (90.0, 1.0) },
            });
            ds.SetColumn("Name", new object?[] { "a", "b" });
            ds.SetColumn("Score", new object?[] { 0.5, null });
            ds.SetColumn("Count", new object?[] { 1, 2 });
            ds.FillColumn("Flag", true);
            ds.SetValue(1, "Name", "renamed");
            ds.Description = "built from C#";
            ds.SetAttribute("source", "unit test");
            Assert.True(ds.AddTag("csharp"));
            Assert.False(ds.AddTag("csharp"));
            Assert.True(ds.SetSpecId(prefix: "SP"));
            ds.SetPeakColumn("formula", new object?[] { "X", "", "Y" });
            ds.Columns = new[] { "Name", "Score", "Count", "Flag", "SpecID" };

            string file = Path.Combine(tmp, "built.msds");
            ds.Save(file);
            using (var back = Dataset.Load(file))
            {
                Assert.Equal(new object?[] { "a", "renamed" }, back["Name"]);
                Assert.Equal(new object?[] { 0.5, null }, back["Score"]);
                Assert.Equal(new object?[] { 1L, 2L }, back["Count"]);
                Assert.Equal(new object?[] { true, true }, back["Flag"]);
                Assert.Equal(new object?[] { "SP1", "SP2" }, back["SpecID"]);
                Assert.Equal("built from C#", back.Description);
                Assert.Equal(new[] { new KeyValuePair<string, string>("source", "unit test") }, back.Attributes);
                Assert.Equal(new[] { "csharp" }, back.Tags);
                Assert.Equal(new object?[] { "X", "", "Y" }, back.GetPeakColumn("formula"));
            }
            foreach (var ext in new[] { "msp", "mgf", "tsv", "csv" })
            {
                string output = Path.Combine(tmp, $"built.{ext}");
                ds.Save(output);
                Assert.Equal(2, Dataset.Load(output).Count);
            }

            using var joined = Dataset.Concat(Dataset.Load(Example), ds);
            Assert.Equal(4, joined.Count);
            Assert.Null(joined.GetValue(0, "Score"));
        }
        finally
        {
            Directory.Delete(tmp, recursive: true);
        }
    }

    [Fact]
    public void BytesRoundTrip()
    {
        using var ds = Dataset.Load(Example).Select(1);
        byte[] data = ds.ToBytes();
        Assert.Equal("HDF", System.Text.Encoding.ASCII.GetString(data, 1, 3));
        using var back = Dataset.FromBytes(data);
        Assert.Equal(new object?[] { "MassSpecID0000002" }, back["Name"]);
        Assert.Throws<MsEntityFormatException>(() => Dataset.FromBytes(System.Text.Encoding.ASCII.GetBytes("not an hdf5 file")));
    }

    [Fact]
    public void ReportsVersion() => Assert.Equal("0.1.0", Dataset.Version);
}
