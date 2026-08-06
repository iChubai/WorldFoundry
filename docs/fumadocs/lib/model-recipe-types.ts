export type ModelRecipeStatusGroup =
  | 'verified'
  | 'integrated'
  | 'runtime_ported'
  | 'profile'
  | 'planned'
  | 'blocked';

export type ModelRecipeStatus = {
  group: ModelRecipeStatusGroup;
  label: string;
  integration: string;
  runner: string;
  demo: string;
};

export type ModelRecipeCategory = {
  id: string;
  label: string;
  label_zh: string;
  description: string;
  count: number;
};

export type ModelRecipeRuntimeSummary = {
  profileId: string | null;
  environmentName: string | null;
  environmentKind: 'dedicated' | 'unified' | 'unrecorded';
  python: string | null;
  cudaLabel: string | null;
};

export type ModelRecipeIndexEntry = {
  id: string;
  name: string;
  category: string;
  categoryLabel: string;
  categoryLabelZh: string;
  provider: string;
  summary: string;
  aliases: string[];
  tasks: string[];
  status: ModelRecipeStatus;
  runtime: ModelRecipeRuntimeSummary;
  checkpoint: {
    id: string;
    revision?: string;
    license?: string;
    gated?: boolean;
    private?: boolean;
    role?: string;
    status?: string;
  } | null;
};

export type ModelRecipeIndexData = {
  total: number;
  categories: ModelRecipeCategory[];
  recipes: ModelRecipeIndexEntry[];
};

export type ModelRecipeSource = {
  kind: 'project' | 'paper' | 'docs' | 'source' | 'weights';
  label: string;
  url: string;
  revision?: string;
};

export type ModelRecipeContractField = { field: string; detail: string };
export type ModelRecipeArtifact = { kind: string; filename: string };

export type ModelRecipeTaskField = {
  field: string;
  detail: string;
  kind?: string;
  target?: string;
  required?: boolean;
  default?: unknown;
  choices?: string[];
  description?: string;
};

export type ModelRecipeTaskArtifact = {
  kind: string;
  filename: string;
  description?: string;
};

export type ModelRecipeTask = {
  id: string;
  label: string;
  description: string;
  source: 'inference_spec' | 'catalog';
  variantIds: string[];
  inputs: ModelRecipeTaskField[];
  artifacts: ModelRecipeTaskArtifact[];
};

export type ModelRecipeCheckpoint = {
  id: string;
  revision?: string;
  license?: string;
  gated?: boolean;
  private?: boolean;
  role?: string;
  status?: string;
  notes?: string[];
};

export type ModelRecipeVariant = {
  id: string;
  label: string;
  task: string;
  runtimeProfile: string;
  pipelineBinding: string;
  status: string;
  pipelineTarget: string | null;
  runner: string | null;
  loadingMethod: string | null;
  invocationMode: string | null;
  environmentName: string | null;
  environmentKind: 'dedicated' | 'unified' | 'unrecorded';
  python: string | null;
  cudaLabel: string | null;
  backendStage: string | null;
  runtimeStatus: string | null;
  inputContract: ModelRecipeContractField[];
  artifacts: ModelRecipeArtifact[];
};

export type ModelRecipeRuntime = ModelRecipeRuntimeSummary & {
  bindingId: string | null;
  runnerTarget: string | null;
  runner: string | null;
  pipelineTarget: string | null;
  loadingMethod: string | null;
  invocationMode: string | null;
  backendStage: string | null;
  runtimeStatus: string | null;
  environmentId: string | null;
  cudaProfile: string | null;
  driverStatus: string | null;
  condaPackages: string[];
  pipPackages: string[];
  packageVersions: Record<string, string>;
  validationImports: string[];
  notes: string[];
};

export type ModelRecipeOutputSpec = {
  label: string;
  detail: string;
};

// Manifest-derived facts that the generator now surfaces explicitly instead of
// leaving buried in the raw YAML. Every field is optional: missing stays
// missing rather than being inferred into a stronger claim.
export type ModelRecipeFacts = {
  minVramGb: number | null;
  family: string | null;
  developer: string | null;
  releaseType: string | null;
  modalities: string[];
  architectureComponents: string[];
  checkpointKeyValidation: Record<string, number>;
  outputSpecs: ModelRecipeOutputSpec[];
};

export type LocalizedText = { en: string; zh: string };
export type LocalizedList = { en: string[]; zh: string[] };

// Hand-authored editorial layer, merged in from lib/model-editorial/<id>.json.
// The generator only reads and merges these files; it never rewrites them, so
// re-running api:generate preserves the prose.
export type ModelRecipeEditorial = {
  tagline?: LocalizedText;
  overview?: LocalizedList;
  whenToUse?: LocalizedText;
  notFor?: LocalizedText;
  highlights?: LocalizedList;
};

export type ModelRecipe = Omit<ModelRecipeIndexEntry, 'runtime' | 'checkpoint'> & {
  runtime: ModelRecipeRuntime;
  sources: ModelRecipeSource[];
  checkpoints: ModelRecipeCheckpoint[];
  variants: ModelRecipeVariant[];
  inferenceTasks: ModelRecipeTask[];
  inputContract: ModelRecipeContractField[];
  artifacts: ModelRecipeArtifact[];
  notes: string[];
  facts: ModelRecipeFacts;
  editorial?: ModelRecipeEditorial;
  commands: {
    prepare: string;
    install: string;
    inspect: string;
    check: string;
    run: string;
  };
  catalogPath: string;
};

export type ModelRecipeData = {
  total: number;
  categories: ModelRecipeCategory[];
  recipes: ModelRecipe[];
};
