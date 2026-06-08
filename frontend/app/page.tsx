"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  ApiError,
  checkBackendHealth,
  evaluateSequence,
  type EvaluateResponse,
} from "@/lib/api";

/** Example sequence for quick smoke-testing the pipeline. */
const DEMO_SEQUENCE =
  "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";

export default function HomePage() {
  const [dnaSequence, setDnaSequence] = useState("");
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  // Poll backend health on mount so the status dot reflects connectivity.
  useEffect(() => {
    checkBackendHealth().then(setBackendOnline);
  }, []);

  const handleSubmit = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      setError(null);
      setResult(null);
      setIsLoading(true);

      try {
        const response = await evaluateSequence({
          dna_sequence: dnaSequence.trim(),
        });
        setResult(response);
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err.message);
        } else {
          setError("Failed to reach the orchestrator. Is the backend running?");
        }
      } finally {
        setIsLoading(false);
      }
    },
    [dnaSequence],
  );

  return (
    <main className="min-h-screen bg-cyber-bg text-slate-200">
      {/* Ambient gradient orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-32 top-0 h-96 w-96 rounded-full bg-cyber-cyan/5 blur-3xl" />
        <div className="absolute -right-32 bottom-0 h-96 w-96 rounded-full bg-cyber-magenta/5 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-4xl px-6 py-12">
        {/* Header */}
        <header className="mb-10 text-center">
          <div className="mb-2 flex items-center justify-center gap-3">
            <StatusDot online={backendOnline} />
            <span className="text-xs uppercase tracking-widest text-cyber-muted">
              {backendOnline === null
                ? "Checking backend…"
                : backendOnline
                  ? "Orchestrator online"
                  : "Backend offline"}
            </span>
          </div>
          <h1 className="font-display text-4xl font-black uppercase tracking-wider text-glow-cyan text-cyber-cyan md:text-5xl">
            DeepGenomic
          </h1>
          <p className="mt-2 font-display text-sm uppercase tracking-[0.3em] text-cyber-magenta text-glow-magenta">
            Orchestrator
          </p>
          <p className="mt-4 text-sm text-cyber-muted">
            Local-first · Privacy-focused · LangGraph pipeline
          </p>
        </header>

        {/* Input panel */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="rounded-lg border border-cyber-border bg-cyber-panel/80 p-6 shadow-neon backdrop-blur-sm">
            <label
              htmlFor="dna-sequence"
              className="mb-3 block text-xs font-bold uppercase tracking-widest text-cyber-cyan"
            >
              DNA Sequence
            </label>
            <textarea
              id="dna-sequence"
              value={dnaSequence}
              onChange={(e) => setDnaSequence(e.target.value)}
              placeholder="Paste your DNA sequence here (A, T, C, G, N)…"
              rows={8}
              className="w-full resize-y rounded border border-cyber-border bg-cyber-bg/60 px-4 py-3 font-mono text-sm text-cyber-green placeholder:text-cyber-muted/50 focus:border-cyber-cyan focus:outline-none focus:ring-1 focus:ring-cyber-cyan/50"
              spellCheck={false}
            />
            <div className="mt-2 flex items-center justify-between text-xs text-cyber-muted">
              <span>{dnaSequence.length} bp</span>
              <button
                type="button"
                onClick={() => setDnaSequence(DEMO_SEQUENCE)}
                className="text-cyber-cyan/70 transition hover:text-cyber-cyan"
              >
                Load demo sequence
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || !dnaSequence.trim()}
            className="group relative w-full overflow-hidden rounded-lg border border-cyber-cyan/50 bg-cyber-cyan/10 px-8 py-4 font-display text-sm font-bold uppercase tracking-widest text-cyber-cyan transition hover:bg-cyber-cyan/20 hover:shadow-neon disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <Spinner />
                Running pipeline…
              </span>
            ) : (
              "Evaluate Sequence"
            )}
          </button>
        </form>

        {/* Error display */}
        {error && (
          <div className="mt-6 rounded-lg border border-red-500/40 bg-red-950/30 px-5 py-4 text-sm text-red-300">
            <span className="font-bold uppercase tracking-wider">Error · </span>
            {error}
          </div>
        )}

        {/* Results panel */}
        {result && (
          <section className="mt-8 space-y-4">
            <h2 className="font-display text-xs font-bold uppercase tracking-widest text-cyber-magenta">
              Pipeline Results
            </h2>

            <ResultCard title="Final Evaluation" accent="cyan">
              <pre className="whitespace-pre-wrap text-sm leading-relaxed text-slate-300">
                {result.final_evaluation}
              </pre>
            </ResultCard>

            <div className="grid gap-4 md:grid-cols-2">
              <ResultCard title="Cas-OFFinder" accent="magenta">
                {result.cas_offinder_result ? (
                  <div className="space-y-2 text-sm text-slate-400">
                    {result.cas_offinder_result.error ? (
                      <p className="text-red-300">
                        {result.cas_offinder_result.error}
                      </p>
                    ) : (
                      <p>
                        {result.cas_offinder_result.hit_count} off-target
                        hit(s) found (threshold:{" "}
                        {result.cas_offinder_result.mismatch_threshold})
                      </p>
                    )}
                    <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-cyber-bg/60 p-3 text-xs text-cyber-green">
                      {JSON.stringify(result.cas_offinder_result, null, 2)}
                    </pre>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">No results</p>
                )}
              </ResultCard>
              <ResultCard title="HyenaDNA Score" accent="green">
                <p className="text-sm text-slate-400">{result.hyenadna_score}</p>
              </ResultCard>
            </div>

            <p className="text-center text-xs text-cyber-muted">
              Step: {result.current_step} · {result.input_sequence.length} bp
              processed
            </p>
          </section>
        )}
      </div>
    </main>
  );
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                      */
/* ------------------------------------------------------------------ */

function StatusDot({ online }: { online: boolean | null }) {
  const color =
    online === null
      ? "bg-yellow-400 animate-pulse"
      : online
        ? "bg-cyber-green shadow-[0_0_8px_#39ff14]"
        : "bg-red-500 shadow-[0_0_8px_#ef4444]";

  return <span className={`inline-block h-2 w-2 rounded-full ${color}`} />;
}

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function ResultCard({
  title,
  accent,
  children,
}: {
  title: string;
  accent: "cyan" | "magenta" | "green";
  children: React.ReactNode;
}) {
  const borderColor = {
    cyan: "border-cyber-cyan/30",
    magenta: "border-cyber-magenta/30",
    green: "border-cyber-green/30",
  }[accent];

  const titleColor = {
    cyan: "text-cyber-cyan",
    magenta: "text-cyber-magenta",
    green: "text-cyber-green",
  }[accent];

  return (
    <div
      className={`rounded-lg border ${borderColor} bg-cyber-panel/60 p-5 backdrop-blur-sm`}
    >
      <h3
        className={`mb-3 text-xs font-bold uppercase tracking-widest ${titleColor}`}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}
