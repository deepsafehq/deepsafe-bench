"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppNav } from "@/components/app-nav";
import { useAuth } from "@/components/auth-provider";
import { AuthDisabledNotice } from "@/components/auth-disabled-notice";
import {
  Activity,
  Zap,
  AlertTriangle,
  Clock,
  Key,
  FileText,
  Mail,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

import { API_URL } from "@/lib/api";
const PAGE_SIZE = 10;

interface Detection {
  id: string;
  media_type: string;
  verdict: string;
  confidence: number;
  created_at: string;
}

interface PlanInfo {
  tier: string;
  scans_used: number;
  scans_limit: number;
  is_monthly: boolean;
}

function KpiCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-surface border border-border rounded-md p-4 sm:p-5">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-4 h-4 text-accent" />
        <span className="text-[13px] text-text-secondary">{label}</span>
      </div>
      <div className="font-heading text-[28px] font-light text-text-primary">
        {value}
      </div>
    </div>
  );
}

const MEDIA_FILTERS = ["all", "image", "audio", "video"] as const;
type MediaFilter = (typeof MEDIA_FILTERS)[number];

export default function DashboardPage() {
  const { session, isLoading: authLoading, user , supabase } = useAuth();
  // No Supabase client means auth is not configured for this build.
  const authConfigured = Boolean(supabase);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [plan, setPlan] = useState<PlanInfo | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [totalDetections, setTotalDetections] = useState(0);
  const [page, setPage] = useState(0);
  const [mediaFilter, setMediaFilter] = useState<MediaFilter>("all");

  const fetchDashboard = useCallback(
    async (pageNum: number, filter: MediaFilter) => {
      if (!session?.access_token) return;

      try {
        const params = new URLSearchParams({
          offset: String(pageNum * PAGE_SIZE),
          limit: String(PAGE_SIZE),
        });
        if (filter !== "all") params.set("media_type", filter);

        const res = await fetch(`${API_URL}/api/dashboard?${params}`, {
          headers: { Authorization: `Bearer ${session.access_token}` },
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => null);
          setFetchError(
            errData?.detail ||
              `Failed to load dashboard (${res.status}). Please refresh.`,
          );
        } else {
          const data = await res.json();
          setFetchError(null);
          setPlan(data.plan);
          setDetections(data.detections);
          setTotalDetections(data.total_detections);
        }
      } catch (e) {
        console.error("Failed to fetch dashboard", e);
        setFetchError(
          "Network error. Please check your connection and refresh.",
        );
      } finally {
        setLoading(false);
      }
    },
    [session],
  );

  useEffect(() => {
    if (authLoading || !user) return;
    fetchDashboard(page, mediaFilter);
  }, [authLoading, user, fetchDashboard, page, mediaFilter]);

  const handleFilterChange = (filter: MediaFilter) => {
    setMediaFilter(filter);
    setPage(0);
  };

  const totalPages = Math.max(1, Math.ceil(totalDetections / PAGE_SIZE));
  const usagePercent = plan
    ? Math.min((plan.scans_used / plan.scans_limit) * 100, 100)
    : 0;
  // scans_limit is null on an unmetered (self-hosted) plan, which is not zero.
  const isUnmetered = !plan || plan.scans_limit == null;
  const scansRemaining = isUnmetered
    ? null
    : (plan!.scans_limit as number) - plan!.scans_used;
  const usedFraction =
    isUnmetered || !plan!.scans_limit
      ? 0
      : (scansRemaining as number) / (plan!.scans_limit as number);
  const remainingColor = isUnmetered
    ? "#34d399"
    : usedFraction > 0.5
      ? "#34d399"
      : usedFraction > 0.2
        ? "#fb923c"
        : "#f87171";

  if (!authConfigured) {
    return <AuthDisabledNotice page="The dashboard" />;
  }

  return (
    <div
      data-theme="app"
      className="min-h-screen bg-background text-text-primary"
    >
      <AppNav />
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
        {loading ? (
          <p
            role="status"
            aria-live="polite"
            className="text-sm text-text-tertiary"
          >
            Loading...
          </p>
        ) : fetchError ? (
          <div
            role="alert"
            className="px-4 py-3 rounded-md bg-danger-light border border-danger/20 text-sm text-danger"
          >
            {fetchError}
          </div>
        ) : (
          <div aria-live="polite" className="space-y-6">
            <h1 className="text-2xl font-heading font-light text-text-primary">
              Dashboard
            </h1>

            {/* KPI Row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
              <KpiCard
                icon={Activity}
                label="Scans Used"
                value={String(plan?.scans_used ?? 0)}
              />
              <KpiCard
                icon={Zap}
                label="Remaining"
                value={scansRemaining === null ? "Unlimited" : String(scansRemaining)}
                color={remainingColor}
              />
              <KpiCard
                icon={AlertTriangle}
                label="Total Detections"
                value={String(totalDetections)}
              />
              <KpiCard
                icon={Clock}
                label="Plan"
                value={
                  plan?.tier === "free"
                    ? "Free"
                    : plan?.tier === "starter"
                      ? "Starter"
                      : "Pro"
                }
              />
            </div>

            {/* Usage Bar */}
            <div className="bg-surface border border-border rounded-md p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-text-secondary">
                  {plan?.scans_used} / {plan?.scans_limit} scans
                  {plan?.is_monthly ? " this month" : ""}
                </span>
                {plan?.tier === "free" && (
                  <a
                    href="https://github.com/deepsafehq/deepsafe-bench"
                    className="px-3 py-1.5 text-xs font-medium bg-accent text-accent-foreground rounded-md hover:bg-accent-hover transition-colors"
                  >
                    Upgrade
                  </a>
                )}
              </div>
              <div className="w-full h-2.5 bg-surface-raised rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    usagePercent > 90
                      ? "bg-danger"
                      : usagePercent > 70
                        ? "bg-warning"
                        : "bg-accent"
                  }`}
                  style={{ width: `${usagePercent}%` }}
                />
              </div>
              {!plan?.is_monthly && plan?.tier === "free" && (
                <p className="text-xs text-text-tertiary mt-2">
                  Free tier includes {plan?.scans_limit} lifetime scans.
                </p>
              )}
            </div>

            {/* Detections */}
            <div>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <h2 className="text-lg font-heading font-light text-text-primary">
                  Detection History
                </h2>
                {/* Media type filter */}
                <div className="flex items-center gap-1 bg-surface border border-border rounded-md p-1">
                  {MEDIA_FILTERS.map((f) => (
                    <button
                      key={f}
                      onClick={() => handleFilterChange(f)}
                      className={`px-3 py-1.5 text-xs font-medium rounded-sm transition-colors capitalize ${
                        mediaFilter === f
                          ? "bg-accent text-accent-foreground"
                          : "text-text-secondary hover:text-text-primary hover:bg-surface-raised"
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>

              {detections.length === 0 ? (
                <div className="border border-dashed border-border rounded-md p-12 text-center">
                  <Activity className="w-8 h-8 mx-auto text-text-tertiary mb-3" />
                  <p className="text-sm text-text-tertiary">
                    {mediaFilter !== "all"
                      ? `No ${mediaFilter} detections found.`
                      : "No detections yet. Try analyzing a file."}
                  </p>
                  {mediaFilter === "all" && (
                    <a
                      href="/"
                      className="text-xs text-accent hover:text-accent-hover mt-1 inline-block"
                    >
                      Run your first scan
                    </a>
                  )}
                </div>
              ) : (
                <>
                  {/* Desktop table */}
                  <div className="hidden sm:block border border-border rounded-md overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border-subtle bg-surface">
                          <th className="text-left px-4 py-3 text-text-secondary text-sm font-medium">
                            ID
                          </th>
                          <th className="text-left px-4 py-3 text-text-secondary text-sm font-medium">
                            Date
                          </th>
                          <th className="text-left px-4 py-3 text-text-secondary text-sm font-medium">
                            Type
                          </th>
                          <th className="text-left px-4 py-3 text-text-secondary text-sm font-medium">
                            Verdict
                          </th>
                          <th className="text-left px-4 py-3 text-text-secondary text-sm font-medium">
                            Confidence
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {detections.map((d, i) => (
                          <tr
                            key={d.id}
                            className={`border-b border-border-subtle ${i % 2 === 1 ? "bg-surface" : "bg-background"}`}
                          >
                            <td className="px-4 py-3 font-mono text-xs text-text-tertiary">
                              {d.id}
                            </td>
                            <td className="px-4 py-3 text-text-primary">
                              {new Date(d.created_at).toLocaleDateString(
                                "en-US",
                                {
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                },
                              )}
                            </td>
                            <td className="px-4 py-3">
                              <span className="bg-accent-light text-accent rounded-full text-xs px-2 py-0.5 uppercase">
                                {d.media_type}
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <span
                                className={`font-medium ${d.verdict === "fake" ? "text-danger" : "text-accent"}`}
                              >
                                {d.verdict.toUpperCase()}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-text-primary">
                              {(d.confidence * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Mobile cards */}
                  <div className="sm:hidden space-y-2">
                    {detections.map((d) => (
                      <div
                        key={d.id}
                        className="bg-surface border border-border rounded-md p-4"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span
                            className={`font-medium text-sm ${d.verdict === "fake" ? "text-danger" : "text-accent"}`}
                          >
                            {d.verdict.toUpperCase()}
                          </span>
                          <span className="bg-accent-light text-accent rounded-full text-xs px-2 py-0.5 uppercase">
                            {d.media_type}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-xs text-text-tertiary">
                          <span>
                            {new Date(d.created_at).toLocaleDateString(
                              "en-US",
                              {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              },
                            )}
                          </span>
                          <span>{(d.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="text-[10px] font-mono text-text-tertiary mt-1">
                          {d.id}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between mt-4">
                      <button
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={page === 0}
                        className="flex items-center gap-1 px-3 py-2 text-sm text-text-secondary border border-border rounded-md hover:bg-surface disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                      >
                        <ChevronLeft className="w-4 h-4" /> Previous
                      </button>
                      <span className="text-sm text-text-tertiary">
                        Page {page + 1} of {totalPages}
                      </span>
                      <button
                        onClick={() =>
                          setPage((p) => Math.min(totalPages - 1, p + 1))
                        }
                        disabled={page >= totalPages - 1}
                        className="flex items-center gap-1 px-3 py-2 text-sm text-text-secondary border border-border rounded-md hover:bg-surface disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                      >
                        Next <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Quick Links */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <Link
                href="/api-keys"
                className="flex items-center gap-3 p-4 bg-surface border border-border rounded-md hover:border-accent/30 transition-colors"
              >
                <Key className="w-5 h-5 text-accent" />
                <span className="text-sm text-text-primary">API Keys</span>
                <ArrowUpRight className="w-4 h-4 text-text-tertiary ml-auto" />
              </Link>
              <a
                href="https://deepsafehq.github.io/deepsafe-bench/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 p-4 bg-surface border border-border rounded-md hover:border-accent/30 transition-colors"
              >
                <FileText className="w-5 h-5 text-accent" />
                <span className="text-sm text-text-primary">Documentation</span>
                <ArrowUpRight className="w-4 h-4 text-text-tertiary ml-auto" />
              </a>
              <a
                href="https://github.com/deepsafehq/deepsafe-bench/issues"
                className="flex items-center gap-3 p-4 bg-surface border border-border rounded-md hover:border-accent/30 transition-colors"
              >
                <Mail className="w-5 h-5 text-accent" />
                <span className="text-sm text-text-primary">Support</span>
                <ArrowUpRight className="w-4 h-4 text-text-tertiary ml-auto" />
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
