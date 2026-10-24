export type Predictor = "edge_directed" | "bilinear" | "med" | "rcan_style" | "auto";
export type SelectedPredictor = "edge_directed" | "bilinear" | "med" | "rcan_style";

export type PredictorTrial = {
  method: SelectedPredictor;
  bit_length: number;
  mean_abs_residual: number;
  order: number;
};


export type QualityMetrics = {
  psnr: number | string;
  ssim: number;
  mse: number;
  mae: number;
};

export type AttackResult = {
  attack: string;
  domain: string;
  detected: boolean;
  validIndexes: number[];
  usedIndexes: number[];
  recovery: string;
  exact: boolean;
  message: string;
};

export type Rect = {
  top: number;
  left: number;
  height: number;
  width: number;
};

export type Share = {
  index: number;
  label: string;
  image: string;
  valid: boolean;
};

export type MetricRow = {
  share: number;
  entropy: number;
  horizontalCorrelation: number;
  verticalCorrelation: number;
  npcr: number;
  uaci: number;
};

export type Job = {
  jobId: string;
  predictor: SelectedPredictor;
  requestedPredictor: Predictor;
  predictorSelection: "manual" | "automatic";
  predictorCandidates: PredictorTrial[];
  original: string;
  masked: string;
  quality: {
    masked: { whole: QualityMetrics; msa: QualityMetrics | null };
    stage3: QualityMetrics;
  };
  lr: string;
  shares: Share[];
  stats: Record<string, unknown>;
  msaMode: "manual" | "auto_texture" | "auto_edge";
  inpaintingMethod?: "telea" | "contextual";
  msaRect: Rect | null;
  metrics: MetricRow[];
};

export type Recovery = {
  stage: number;
  title: string;
  image: string;
  exact: boolean;
  width: number;
  height: number;
  validIndexes?: number[];
  usedIndexes?: number[];
};
