"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { LandingNavbar } from "@/components/landing/navbar";
import { Footer } from "@/components/landing/footer";

interface ModelRow {
  name: string;
  modality: string;
  n: number;
  auc: number | null;
  kind: "detector" | "provenance";
}

interface GeneratorRow {
  generator: string;
  modality: string;
  n: number;
  caught: number;
  models: Record<string, number>;
}

interface Benchmark {
  samples: number;
  generatorCount: number;
  summary: {
    recall: number;
    fpr: number;
    ensembles: Record<string, { n: number; auc: number | null }>;
  };
  models: ModelRow[];
  generators: GeneratorRow[];
}

type Modality = "all" | "images" | "audio" | "video";

const MODALITIES: Modality[] = ["all", "images", "audio", "video"];

// Groups smaller than this are hidden by default. 325 of the 391 groups are
// MLAAD text-to-speech systems with 8 samples each; at that size a couple of
// coin flips moves a group from 100% to 0% and swamps the ranking. They are
// not dropped, only collapsed behind a toggle, because the data is real: in
// aggregate those groups are caught 73.7% of the time, better than the large
// ones.
const MIN_SAMPLES = 10;

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Colour by detection rate: this is the signal, so it should read instantly. */
function rateColor(rate: number): string {
  if (rate < 0.25) return "var(--danger)";
  if (rate < 0.6) return "#d97706";
  return "var(--accent)";
}

export default function BenchmarkPage() {
  const [data, setData] = useState<Benchmark | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modality, setModality] = useState<Modality>("all");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [includeSmall, setIncludeSmall] = useState(false);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH || "";
    fetch(`${base}/benchmark.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((raw) =>
        setData({
          samples: raw.samples,
          generatorCount: raw.generator_count,
          summary: raw.summary,
          models: raw.models,
          generators: raw.generators,
        }),
      )
      .catch((e) => setError(String(e)));
  }, []);

  const generators = useMemo(() => {
    if (!data) return [];
    const needle = query.trim().toLowerCase();
    return data.generators.filter(
      (g) =>
        (modality === "all" || g.modality === modality) &&
        (includeSmall || g.n >= MIN_SAMPLES) &&
        (!needle || g.generator.toLowerCase().includes(needle)),
    );
  }, [data, modality, query, includeSmall]);

  const smallCount = useMemo(
    () => (data ? data.generators.filter((g) => g.n < MIN_SAMPLES).length : 0),
    [data],
  );

  if (error) {
    return (
      <Shell>
        <p className="text-sm text-danger">
          Could not load benchmark data: {error}
        </p>
      </Shell>
    );
  }

  if (!data) {
    return (
      <Shell>
        <p className="text-sm text-text-secondary">Loading benchmark data...</p>
      </Shell>
    );
  }

  return (
    <Shell>
      <header className="mb-12">
        <h1 className="text-3xl font-medium text-text-primary mb-3">
          Benchmark explorer
        </h1>
        <p className="text-text-secondary max-w-2xl leading-relaxed">
          Every number here is computed from {data.samples.toLocaleString()}{" "}
          evaluated samples across {data.generatorCount} generators, and is
          reproducible from the prediction matrix shipped in the repository.
        </p>
      </header>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border-subtle border border-border-subtle mb-4">
        <Stat label="Recall on fakes" value={pct(data.summary.recall)} emphasis />
        <Stat label="False positive rate" value={pct(data.summary.fpr)} />
        <Stat
          label="Image ensemble AUC"
          value={data.summary.ensembles.images?.auc?.toFixed(4) ?? "n/a"}
        />
        <Stat
          label="Video ensemble AUC"
          value={data.summary.ensembles.video?.auc?.toFixed(4) ?? "n/a"}
          emphasis
        />
      </section>

      <p className="text-sm text-text-secondary mb-12 max-w-2xl leading-relaxed">
        A third of fake media is missed outright. The video ensemble sits barely
        above chance once it meets a generator it was not trained on, which is
        the finding this project exists to publish.
      </p>

      <section className="mb-16">
        <div className="flex flex-wrap items-center gap-3 mb-6">
          <h2 className="text-lg font-medium text-text-primary mr-auto">
            Detection rate by generator
          </h2>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter generators"
            aria-label="Filter generators"
            className="px-3 py-1.5 text-sm bg-card border border-border-subtle text-text-primary placeholder:text-text-secondary focus:outline-none focus:border-accent"
          />
          <div className="flex border border-border-subtle" role="group">
            {MODALITIES.map((m) => (
              <button
                key={m}
                onClick={() => setModality(m)}
                aria-pressed={modality === m}
                className={`px-3 py-1.5 text-sm transition-colors ${
                  modality === m
                    ? "bg-accent text-white"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-baseline justify-between gap-3 mb-4">
          <p className="text-sm text-text-secondary">
            {generators.length} generator groups, worst first. Click one to see
            what each model scored.
          </p>
          <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
            <input
              type="checkbox"
              checked={includeSmall}
              onChange={(e) => setIncludeSmall(e.target.checked)}
              className="accent-[var(--accent)]"
            />
            Include {smallCount} groups with fewer than {MIN_SAMPLES} samples
          </label>
        </div>
        {includeSmall && (
          <p className="text-xs text-text-secondary mb-4 max-w-2xl">
            Small groups are mostly MLAAD text-to-speech systems with 8 samples
            each. Individually they are too small to rank on, though in
            aggregate they are caught 73.7% of the time.
          </p>
        )}

        <ul className="border border-border-subtle divide-y divide-border-subtle">
          {generators.map((g) => {
            const key = `${g.modality}/${g.generator}`;
            const open = expanded === key;
            return (
              <li key={key}>
                <button
                  onClick={() => setExpanded(open ? null : key)}
                  aria-expanded={open}
                  className="w-full flex items-center gap-4 px-4 py-2.5 text-left hover:bg-card transition-colors"
                >
                  <span className="w-44 shrink-0 text-sm text-text-primary truncate font-mono">
                    {g.generator}
                  </span>
                  <span className="w-16 shrink-0 text-xs text-text-secondary">
                    {g.modality}
                  </span>
                  <span className="flex-1 h-2 bg-border-subtle relative overflow-hidden">
                    <span
                      className="absolute inset-y-0 left-0"
                      style={{
                        width: `${Math.max(g.caught * 100, 0.7)}%`,
                        backgroundColor: rateColor(g.caught),
                      }}
                    />
                  </span>
                  <span
                    className="w-16 shrink-0 text-sm text-right tabular-nums"
                    style={{ color: rateColor(g.caught) }}
                  >
                    {pct(g.caught)}
                  </span>
                  <span className="w-14 shrink-0 text-xs text-text-secondary text-right tabular-nums">
                    n={g.n}
                  </span>
                </button>

                {open && (
                  <div className="px-4 pb-4 pt-1 bg-card">
                    <p className="text-xs text-text-secondary mb-3">
                      Mean score each model gave this generator. Higher means
                      more suspicious; anything near zero is a model that saw
                      nothing wrong.
                    </p>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-1">
                      {Object.entries(g.models)
                        .sort((a, b) => b[1] - a[1])
                        .map(([name, score]) => (
                          <div
                            key={name}
                            className="flex justify-between text-xs py-0.5 border-b border-border-subtle"
                          >
                            <span className="text-text-secondary font-mono truncate">
                              {name}
                            </span>
                            <span className="tabular-nums text-text-primary ml-2">
                              {score.toFixed(3)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <section className="mb-16">
        <h2 className="text-lg font-medium text-text-primary mb-2">
          Individual models
        </h2>
        <p className="text-sm text-text-secondary mb-6 max-w-2xl">
          AUC over every sample of that model&apos;s modality. The uncomfortable
          line is npr_video, an image detector applied frame by frame, which
          outscores every purpose-built video model here.
        </p>
        <ModelTable rows={data.models.filter((m) => m.kind === "detector")} />

        <h3 className="text-base font-medium text-text-primary mt-10 mb-2">
          Provenance services
        </h3>
        <p className="text-sm text-text-secondary mb-4 max-w-2xl">
          These answer whether a watermark or manifest is present, not whether
          media is fake. AUC is the wrong metric for them and is shown only to
          explain why they are ranked separately: a watermark detector scores at
          chance on media carrying no watermark. When a signal is present, it is
          far stronger evidence than any detector score.
        </p>
        <ModelTable rows={data.models.filter((m) => m.kind === "provenance")} />
      </section>

      <footer className="border-t border-border-subtle pt-6 text-sm text-text-secondary">
        <p className="mb-2">
          Reproduce any of this with{" "}
          <code className="font-mono text-text-primary">
            deepsafe eval --baseline ensemble --modality video
          </code>
          .
        </p>
        <Link href="/" className="text-accent hover:underline">
          Back to overview
        </Link>
      </footer>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background text-text-primary font-sans">
      <LandingNavbar />
      <main className="max-w-[1100px] mx-auto px-6 py-16">{children}</main>
      <Footer />
    </div>
  );
}

function Stat({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div className="bg-background px-4 py-5">
      <div className="text-xs text-text-secondary mb-1.5">{label}</div>
      <div
        className={`text-2xl tabular-nums ${
          emphasis ? "text-danger" : "text-text-primary"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function ModelTable({ rows }: { rows: ModelRow[] }) {
  return (
    <div className="overflow-x-auto border border-border-subtle">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border-subtle text-text-secondary">
            <th className="text-left font-normal px-4 py-2">Model</th>
            <th className="text-left font-normal px-4 py-2">Modality</th>
            <th className="text-right font-normal px-4 py-2">Samples</th>
            <th className="text-right font-normal px-4 py-2">AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.name} className="border-b border-border-subtle last:border-0">
              <td className="px-4 py-2 font-mono text-text-primary">{m.name}</td>
              <td className="px-4 py-2 text-text-secondary">{m.modality}</td>
              <td className="px-4 py-2 text-right tabular-nums text-text-secondary">
                {m.n.toLocaleString()}
              </td>
              <td
                className="px-4 py-2 text-right tabular-nums"
                style={{ color: m.auc && m.auc < 0.6 ? "var(--danger)" : undefined }}
              >
                {m.auc?.toFixed(4) ?? "n/a"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
