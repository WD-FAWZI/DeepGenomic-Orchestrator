/**
 * API client for the Python FastAPI backend.
 *
 * Centralizes fetch logic so page components stay focused on UI concerns.
 * The evaluate endpoint maps 1:1 to the LangGraph pipeline in backend/agent/.
 */

/** FastAPI backend base URL — must match the running uvicorn port. */
const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export const EVALUATE_ENDPOINT = `${API_BASE}/api/evaluate`;
export const HEALTH_ENDPOINT = `${API_BASE}/health`;

export interface EvaluateRequest {
  dna_sequence: string;
  guide_sequence?: string;
}

export interface CasOffinderHit {
  chromosome: string;
  position: number;
  sequence: string;
  mismatches: number;
  strand?: string;
  query_sequence?: string;
}

export interface CasOffinderResult {
  off_targets: CasOffinderHit[];
  hit_count: number;
  guide_sequence: string;
  genome_path: string;
  mismatch_threshold: number;
  error?: string;
}

export interface BiologicalFilterResult {
  guide_seq: string;
  gc_content: number;
  shannon_entropy: number;
  has_homopolymer: boolean;
  has_polyT_u6: boolean;
  self_comp_max: number;
  viable: boolean;
  reasons: string;
  score: number;
}

export interface EvaluateResponse {
  input_sequence: string;
  current_step: string;
  cas_offinder_result: CasOffinderResult | null;
  hyenadna_score: number | null;
  biological_filter_result: BiologicalFilterResult | null;
  final_evaluation: string;
  metadata: Record<string, unknown>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * POST /api/evaluate — triggers the LangGraph orchestration pipeline.
 */
export async function evaluateSequence(
  payload: EvaluateRequest,
): Promise<EvaluateResponse> {
  const response = await fetch(EVALUATE_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    const detail =
      typeof errorBody.detail === "string"
        ? errorBody.detail
        : `Request failed with status ${response.status}`;
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<EvaluateResponse>;
}

/**
 * GET /health — quick connectivity check for the status indicator.
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(HEALTH_ENDPOINT, {
      method: "GET",
      signal: AbortSignal.timeout(3000),
    });
    return response.ok;
  } catch {
    return false;
  }
}
