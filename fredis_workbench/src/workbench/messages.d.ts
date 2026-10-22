export interface JobInfo {
  id: string; type: 'training' | 'preparation' | 'prediction' | 'batch-prediction'; name: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  pid?: number; startedAt: string; finishedAt?: string;
  outputDir: string; logPath: string; command: string[]; epochs?: number; detached?: boolean; resultsPath?: string;
  configurationId?: string; projectRoot?: string | null; warning?: string; trackingError?: string;
  error?: string; exitCode?: number | null;
}
export type WorkbenchRequest =
  | { type: 'workbench/ready' | 'workbench/environment' | 'workbench/settings' | 'workbench/terminal' }
  | { type: 'workbench/stop' | 'workbench/logs' | 'workbench/openOutput' | 'workbench/predictionResult'; jobId: string }
  | { type: 'workbench/modelConfig'; path: string; open: boolean }
  | { type: 'workbench/trainingInspect'; field: 'trainDir' | 'valDir'; path: string }
  | { type: 'workbench/trainingDataset'; field: 'trainDir' | 'valDir'; path: string }
  | { type: 'workbench/saveModelConfig'; json: string };
export type WorkbenchEvent =
  | { type: 'workbench/jobs'; jobs: JobInfo[] }
  | { type: 'workbench/jobSelected'; jobId: string }
  | { type: 'workbench/navigate'; page: string }
  | { type: 'workbench/logs'; job: JobInfo; text: string }
  | { type: 'workbench/trainingDataset'; field: 'trainDir' | 'valDir'; path: string; count?: number; bytes?: number; files?: string[]; summary?: Record<string, unknown>; error?: string }
  | { type: 'workbench/error'; error: string }
  | { type: 'workbench/modelConfig'; config: Record<string, unknown> }
  | { type: 'workbench/environment'; error?: string; data?: { torch: string; cuda: boolean; gpu: string | null; root: string; python: string } };

export type ParameterRequest =
  | { type: 'training/checkpoint'; path: string }
  | { type: 'training/sources'; requestId: string; config: { datasetDir: string; resume?: string } }
  | { type: 'parameters/pick'; target: 'data' | 'training' | 'trainable' }
  | { type: 'parameters/import'; target: 'data' | 'training' | 'trainable'; name: string; json: string }
  | { type: 'parameters/load'; target: 'data' | 'training' | 'trainable'; path: string }
  | { type: 'parameters/save'; target: 'data' | 'trainable'; fragmenterParams: Record<string, unknown> };
export type ParameterEvent =
  | { type: 'parameters/loaded'; target: 'data' | 'training' | 'trainable'; path: string; fragmenterParams: Record<string, unknown> }
  | { type: 'parameters/saved'; target: 'data' | 'trainable'; path: string }
  | { type: 'parameters/error'; target: 'data' | 'training' | 'trainable'; error: string };

export type PreparationDataset = {
  id: string; path: string; split: 'train' | 'validation' | 'test'; mapping?: Record<string, string>;
};
export type PreparationInputs = {
  datasets: PreparationDataset[]; datasetMode: 'assigned' | 'automatic';
  validationRatio?: number; testRatio?: number; validationSeed?: number;
};
export type DataRequest =
  | { type: 'data/columns' | 'data/preview' | 'data/check'; target: 'train' | 'validation' | 'home' | 'prediction' | `dataset-${string}`; requestId: string; path: string; mapping?: Record<string, string>; fragmenterParams?: Record<string, unknown> }
  | { type: 'data/preflight'; requestId: string; config: Record<string, unknown> }
  | { type: 'data/export-splits'; requestId: string; checkedId: string; checkRequestId: string }
  | { type: 'data/model'; requestId: string; path: string };
export type DataProgressEvent = { type: 'data/check-progress'; target: 'train' | 'validation' | 'prediction' | `dataset-${string}`; requestId: string; current: number; total: number };
export type SplitProgressEvent = {
  type: 'data/preflight-progress' | 'data/splits-export-progress'; requestId: string;
  stage: string; percent: number; current?: number; total?: number;
  datasetIndex?: number; datasetsTotal?: number; split?: 'train' | 'validation' | 'test';
};
export type SplitExportEvent =
  | { type: 'data/splits-exported'; requestId: string; outputDir: string; files: Record<string, string> }
  | { type: 'data/splits-export-cancelled'; requestId: string };
export type ResourceRequest =
  | { type: 'library/load' | 'shell/document' | 'shell/github' }
  | { type: 'library/copy' | 'library/open'; path: string }
  | { type: 'library/pick'; target: 'quickTrainDir' | 'quickValDir' | 'quickTrainOutput' | 'quickCheckpoint' | 'quickPredictModel' };

/** Shared with the CLI (`--tree-config`): pattern sets and tree nodes are kept apart. */
export type ReactionPatternTreeConfiguration = {
  patternSets: Array<{ id: string; name: string; patterns: Array<Record<string, unknown>>; source?: string }>;
  nodes: Array<{ id: string; patternSet: string; parent: string | null }>;
};
export type ReactionPatternAnalysisRequest =
  | { type: 'reactionPatternAnalysis/ready' | 'reactionPatternAnalysis/defaults' | 'reactionPatternAnalysis/pickInputs' | 'reactionPatternAnalysis/pickParams' | 'reactionPatternAnalysis/pickOutput' | 'reactionPatternAnalysis/pickResult' | 'reactionPatternAnalysis/stop' }
  | { type: 'reactionPatternAnalysis/pickPatternFiles'; target?: string }
  | { type: 'reactionPatternAnalysis/addSources'; paths: string[] }
  | { type: 'reactionPatternAnalysis/applyPatternFiles'; paths: string[]; target?: string }
  | { type: 'reactionPatternAnalysis/inspectSource'; index: number; path: string; kind?: string }
  | { type: 'reactionPatternAnalysis/loadPatterns' | 'reactionPatternAnalysis/fragmenterPatternSet'; path: string }
  | { type: 'reactionPatternAnalysis/copy' | 'reactionPatternAnalysis/run'; config: Record<string, unknown> & { tree: ReactionPatternTreeConfiguration; extend?: boolean } }
  | { type: 'reactionPatternAnalysis/query'; request: { path: string; action: 'info' | 'nodeSummary' | 'nodeDistribution' | 'scoreFrequency' | 'compoundList' | 'compoundSummary' | 'compoundNodeDistribution' | 'compoundPeaks' | 'peakDetail' | 'compoundReactionCombinations'; nodeRow?: number; filters?: Record<string, unknown>; withoutPrecursor?: boolean } };
export type ReactionPatternAnalysisEvent =
  | { type: 'reactionPatternAnalysis/sourceList'; sources: Array<Record<string, unknown>> }
  | { type: 'reactionPatternAnalysis/sourceInfo'; index: number; source: Record<string, unknown> }
  | { type: 'reactionPatternAnalysis/picked'; field: string; path: string }
  | { type: 'reactionPatternAnalysis/patternSets'; sets: ReactionPatternTreeConfiguration['patternSets']; target?: string | null }
  | { type: 'reactionPatternAnalysis/defaultsLoaded'; defaults: { maxReactionCount: number; massTolerance: string } }
  | { type: 'reactionPatternAnalysis/configLoaded'; path?: string; config: Record<string, unknown>; sources: Array<Record<string, unknown>>; computedNodes?: string[]; extend?: boolean }
  | { type: 'reactionPatternAnalysis/status'; status: 'idle' | 'running' | 'completed' | 'error'; text: string; result?: string; opened?: boolean }
  | { type: 'reactionPatternAnalysis/log'; text: string }
  | { type: 'reactionPatternAnalysis/error'; error: string }
  | { type: 'reactionPatternAnalysis/result'; action: string; result: Record<string, unknown> };

export type DefaultRequest = { type: 'defaults/save'; workflow: 'data' | 'training' | 'prediction'; config: Record<string, unknown> };
export type DefaultEvent = { type: 'defaults/saved' | 'defaults/error'; workflow: 'data' | 'training' | 'prediction'; error?: string };
