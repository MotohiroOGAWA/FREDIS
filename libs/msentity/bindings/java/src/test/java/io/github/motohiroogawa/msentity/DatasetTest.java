package io.github.motohiroogawa.msentity;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DatasetTest {
  static final Path REPO = Path.of(System.getProperty("msentity.repo", "../.."));
  static final Path EXAMPLE = REPO.resolve("docs/examples/example.msp");
  static final Path DATA = REPO.resolve("tests/data");

  @Test
  void loadsTheExample() {
    try (Dataset ds = Dataset.load(EXAMPLE)) {
      assertEquals(2, ds.size());
      assertEquals(7, ds.nPeaksTotal());
      assertEquals(
          List.of("Name", "PrecursorMZ", "AdductType", "Formula", "CollisionEnergy", "NumPeaks"),
          ds.columns());
      assertEquals(List.of("MassSpecID0000001", "MassSpecID0000002"), ds.column("Name"));
      assertEquals("20.0", ds.value(1, "CollisionEnergy"));
      assertTrue(ds.hasColumn("NumPeaks"));
    }
  }

  @Test
  void readsAllFormats() {
    assertEquals(5, Dataset.load(DATA.resolve("sample.msp")).size());
    assertEquals(6, Dataset.load(DATA.resolve("sample.mgf")).size());
    assertEquals(6, Dataset.load(DATA.resolve("sample.msds")).size());
    Dataset withIds = Dataset.load(DATA.resolve("sample.msp"), null, "SP", false);
    assertEquals("SP1", withIds.value(0, "SpecID"));

    String tsv = "Name\tID\tPeak\nA\t007\t1,2\n";
    assertEquals("007", Dataset.readText(tsv, "tsv", false).value(0, "ID"));
    assertEquals(7L, Dataset.readText(tsv, "tsv", true).value(0, "ID"));
  }

  @Test
  void errorsBecomeExceptions() {
    assertThrows(
        MsEntityException.FileAccess.class, () -> Dataset.load(Path.of("/nonexistent/x.msds")));
    Dataset ds = Dataset.load(EXAMPLE);
    assertThrows(MsEntityException.ColumnNotFound.class, () -> ds.column("Missing"));
    assertThrows(MsEntityException.PositionOutOfRange.class, () -> ds.value(5, "Name"));
    assertThrows(
        MsEntityException.InvalidArgument.class, () -> ds.setColumn("Bad", List.of(1, 2, 3)));
    assertThrows(
        MsEntityException.InvalidArgument.class, () -> Dataset.readText("x", "xml", false));
  }

  @Test
  void peaksAndPeakMetadata() {
    Dataset ds = Dataset.load(EXAMPLE);
    assertArrayEquals(new long[] {0, 4, 7}, ds.offsets());
    assertEquals(7, ds.mz().length);
    Dataset.Spectrum first = ds.spectrum(0);
    assertArrayEquals(new double[] {91.0542, 125.0233, 185.0961, 246.1125}, first.mz(), 1e-12);
    assertArrayEquals(new double[] {0.245, 1.0, 0.349, 0.735}, first.intensity(), 1e-12);
    assertEquals(List.of("formula"), ds.peakColumns());
    assertEquals(Arrays.asList("C7H7+", "C6H5O3+", "", "", "C7H7+", "", ""), ds.peakColumn("formula"));

    ds.normalize(100.0);
    ds.sortPeaksByIntensity(false);
    assertArrayEquals(new double[] {100.0, 73.5, 34.9, 24.5}, ds.spectrum(0).intensity(), 1e-9);
    ds.sortPeaksByMz(true);
    assertEquals(91.0542, ds.spectrum(0).mz()[0], 1e-12);
  }

  @Test
  void viewsShareDataAndCopiesDoNot() {
    Dataset ds = Dataset.load(EXAMPLE);
    assertEquals(List.of("MassSpecID0000002"), ds.slice(1, 2).column("Name"));
    assertEquals(List.of("MassSpecID0000002", "MassSpecID0000001"), ds.select(1, 0).column("Name"));
    assertEquals(List.of("MassSpecID0000002"), ds.filter(new boolean[] {false, true}).column("Name"));
    assertEquals("MassSpecID0000002", ds.sortBy("CollisionEnergy", true).value(0, "Name"));

    Dataset view = ds.select(1);
    view.fillColumn("Checked", true);
    assertFalse(ds.hasColumn("Checked"));
    assertEquals(Arrays.asList(null, true), ds.fullView(true).column("Checked"));
    Dataset copy = view.copy();
    copy.fillColumn("Checked", false);
    assertEquals(Arrays.asList(null, true), ds.fullView(true).column("Checked"));
    assertArrayEquals(new long[] {0, 3}, copy.offsets());

    Dataset survivor = ds.select(0);
    ds.close();
    assertEquals(List.of("MassSpecID0000001"), survivor.column("Name"));
  }

  @Test
  void buildEditAndSave(@TempDir Path tmp) {
    Dataset ds =
        Dataset.fromPeaks(new double[][][] {{{100.0, 1.0}, {150.0, 0.5}}, {{90.0, 1.0}}});
    ds.setColumn("Name", List.of("a", "b"));
    ds.setColumn("Score", Arrays.asList(0.5, null));
    ds.setColumn("Count", List.of(1, 2));
    ds.fillColumn("Flag", true);
    ds.setValue(1, "Name", "renamed");
    ds.setDescription("built from Java");
    ds.setAttribute("source", "unit test");
    assertTrue(ds.addTag("java"));
    assertFalse(ds.addTag("java"));
    assertTrue(ds.setSpecId("SpecID", "SP", false, 1));
    ds.setPeakColumn("formula", List.of("X", "", "Y"));
    ds.setColumns(List.of("Name", "Score", "Count", "Flag", "SpecID"));

    Path file = tmp.resolve("built.msds");
    ds.save(file);
    try (Dataset back = Dataset.load(file)) {
      assertEquals(List.of("a", "renamed"), back.column("Name"));
      assertEquals(Arrays.asList(0.5, null), back.column("Score"));
      assertEquals(List.of(1L, 2L), back.column("Count"));
      assertEquals(List.of(true, true), back.column("Flag"));
      assertEquals(List.of("SP1", "SP2"), back.column("SpecID"));
      assertEquals("built from Java", back.description());
      assertEquals(Map.of("source", "unit test"), back.attributes());
      assertEquals(List.of("java"), back.tags());
      assertEquals(List.of("X", "", "Y"), back.peakColumn("formula"));
    }
    for (String ext : List.of("msp", "mgf", "tsv", "csv")) {
      Path out = tmp.resolve("built." + ext);
      ds.save(out);
      assertEquals(2, Dataset.load(out).size(), ext);
    }

    Dataset joined = Dataset.concat(Dataset.load(EXAMPLE), ds);
    assertEquals(4, joined.size());
    assertNull(joined.value(0, "Score"));
  }

  @Test
  void bytesRoundTrip() {
    Dataset ds = Dataset.load(EXAMPLE).select(1);
    byte[] data = ds.toBytes();
    assertEquals("HDF", new String(data, 1, 3, java.nio.charset.StandardCharsets.US_ASCII));
    Dataset back = Dataset.fromBytes(data);
    assertEquals(List.of("MassSpecID0000002"), back.column("Name"));
    assertThrows(
        MsEntityException.Format.class,
        () -> Dataset.fromBytes("not an hdf5 file".getBytes(java.nio.charset.StandardCharsets.US_ASCII)));
  }

  @Test
  void reportsVersion() {
    assertEquals("0.1.0", Dataset.version());
  }
}
