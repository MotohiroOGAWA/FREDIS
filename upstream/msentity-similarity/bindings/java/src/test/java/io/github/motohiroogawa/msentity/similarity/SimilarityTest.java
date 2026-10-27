package io.github.motohiroogawa.msentity.similarity;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.motohiroogawa.msentity.Dataset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SimilarityTest {
  static final Path REPO = Path.of(System.getProperty("mssim.repo", "../.."));
  static final Path MSENTITY_DATA =
      Path.of(System.getProperty("msentity.data", "../../../msentity/tests/data"));
  static final Path SAMPLE = MSENTITY_DATA.resolve("sample.msp");

  // Reference values for sample.msp, cosine, bin width 0.01.
  static final float[] ROW0 = {
    1.0f, 0.7569831609725952f, 0.85833740234375f, 0.6929314732551575f, 0.6837769150733948f
  };

  @Test
  void allPairsAndPairScores() {
    try (Dataset ds = Dataset.load(SAMPLE)) {
      float[][] matrix = Similarity.allPairs(ds, ds);
      assertEquals(5, matrix.length);
      assertArrayEquals(ROW0, matrix[0], 1e-6f);
      float[] scores =
          Similarity.pairScores(
              ds,
              new long[] {0, 2},
              ds,
              new long[] {1, 4},
              Options.defaults().withMethod(Options.Method.REVERSE_COSINE));
      assertArrayEquals(new float[] {0.7586880922317505f, 0.31010058522224426f}, scores, 1e-6f);
      assertThrows(
          SimilarityException.InvalidArgument.class,
          () -> Similarity.allPairs(ds, ds, Options.defaults().withBinWidth(0)));
      assertThrows(
          SimilarityException.PositionOutOfRange.class,
          () -> Similarity.pairScores(ds, new long[] {9}, ds, new long[] {0}));
    }
  }

  @Test
  void librarySearch() {
    try (Dataset ds = Dataset.load(SAMPLE)) {
      AtomicLong last = new AtomicLong();
      try (SimilarityDataset result =
          Similarity.librarySearch(
              ds, ds, 0.9, Options.defaults(), true, "sample.msp", null,
              (done, total) -> last.set(done * 1000 + total))) {
        assertEquals(25_025, last.get());
        assertEquals(9, result.size());
        assertEquals(
            List.of("index1", "index2", "cosine_similarity", "data_index1", "data_index2"),
            result.columns());
        assertEquals(List.of(0L, 1L, 4L, 2L, 3L, 2L, 3L, 1L, 4L), result.column("index2"));
        assertTrue(result.metadataJson().contains("\"mode\":\"library_search\""));
        assertTrue(result.hasMatchedData());
        assertArrayEquals(new long[] {0, 1, 4, 2, 3}, result.sourceIndices(2));
        try (SimilarityDataset.Match match = result.matchRecords(2)) { // pair (1, 4)
          assertEquals(ds.value(1, "Name"), match.first().value(0, "Name"));
          assertEquals(ds.value(4, "Name"), match.second().value(0, "Name"));
          assertArrayEquals(ds.spectrum(4).mz(), match.second().spectrum(0).mz());
        }
      }
    }
  }

  @Test
  void byKey() {
    try (Dataset ds = Dataset.load(SAMPLE)) {
      ds.setSpecId("SpecID", "SP", false, 0);
      try (Dataset reordered = ds.select(4, 3, 2);
          SimilarityDataset result = Similarity.similarityByKey(ds, reordered)) {
        assertEquals(List.of("SpecID", "index1", "index2", "cosine_similarity"), result.columns());
        assertEquals(List.of(2L, 3L, 4L), result.column("index1"));
        assertEquals(List.of(2L, 1L, 0L), result.column("index2"));
        assertFalse(result.hasMatchedData());
        assertThrows(SimilarityException.InvalidArgument.class, () -> result.matchRecords(0));
      }
      assertThrows(
          SimilarityException.ColumnNotFound.class,
          () -> Similarity.similarityByKey(ds, ds, "Missing", "SpecID", Options.defaults()));
    }
  }

  @Test
  void filterSortSaveLoad(@TempDir Path tmp) throws Exception {
    try (Dataset ds = Dataset.load(SAMPLE);
        SimilarityDataset result = Similarity.librarySearch(ds, ds, 0.9)) {
      List<Object> index1 = result.column("index1");
      boolean[] mask = new boolean[index1.size()];
      for (int i = 0; i < mask.length; i++) mask[i] = index1.get(i).equals(1L);
      try (SimilarityDataset filtered = result.filter(mask);
          SimilarityDataset ranked = filtered.sortBy("cosine_similarity", false)) {
        assertEquals(List.of(1L, 4L), ranked.column("index2"));
        Path path = tmp.resolve("hits.mssim");
        ranked.save(path);
        try (SimilarityDataset loaded = SimilarityDataset.load(path)) {
          assertEquals(2, loaded.size());
          assertArrayEquals(new long[] {1, 4}, loaded.sourceIndices(2));
          loaded.exportTable(tmp.resolve("hits.tsv"));
          assertTrue(
              Files.readString(tmp.resolve("hits.tsv")).startsWith("index1\tindex2\tcosine_similarity"));
        }
      }
      assertThrows(
          SimilarityException.FileAccess.class,
          () -> SimilarityDataset.load(tmp.resolve("missing.mssim")));
    }
  }

  @Test
  void referenceFileAndStatistics() {
    try (SimilarityDataset result =
        SimilarityDataset.load(REPO.resolve("tests/data/similarity.mssim"))) {
      assertEquals(16, result.size());
      try (Dataset first = result.matchedDataset(1)) {
        assertEquals(6, first.size());
      }
      SimilarityDataset.ScoreSummary summary = result.describeScores();
      assertEquals(16, summary.count());
      SimilarityDataset.Histogram histogram = result.histogram(4);
      assertEquals(16, java.util.Arrays.stream(histogram.counts()).sum());
      assertEquals(5, histogram.edges().length);
      result.setMetadataJson("{\"note\": \"名前\"}");
      assertEquals("{\"note\": \"名前\"}", result.metadataJson()); // stored as given
      assertThrows(SimilarityException.InvalidArgument.class, () -> result.setMetadataJson("[1]"));
    }
  }
}
