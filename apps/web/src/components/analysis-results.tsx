"use client";

import React from "react";
import { motion } from "framer-motion";
import { AnalysisResults } from "@/hooks/useAnalysis";

interface AnalysisResultsProps {
  results: AnalysisResults;
}

function getConfidenceScore(res: AnalysisResults): string {
  const prob = Number(res.deepfake_probability ?? 0.5);
  const score = res.is_likely_deepfake ? prob : 1 - prob;
  return (score * 100).toFixed(1);
}

export function AnalysisResultsPanel({ results }: AnalysisResultsProps) {
  return (
    <div aria-live="polite" className="w-full max-w-xl space-y-8">
      {/* Verdict Card */}
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        className={`rounded-md p-8 text-center border ${results.is_likely_deepfake ? "bg-danger-light border-danger/20" : "bg-accent-light border-accent/20"}`}
      >
        <div className="space-y-3">
          <h3 className="text-sm font-medium uppercase tracking-widest text-text-secondary">
            Final Verdict
          </h3>
          <div
            className={`font-heading text-[32px] font-normal tracking-tight flex items-center justify-center gap-3 ${results.is_likely_deepfake ? "text-danger" : "text-accent"}`}
          >
            {results.is_likely_deepfake ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="0.75em"
                height="0.75em"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M18 6 6 18" />
                <path d="m6 6 12 12" />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="0.75em"
                height="0.75em"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M20 6 9 17l-5-5" />
              </svg>
            )}
            {results.is_likely_deepfake ? "SYNTHETIC" : "AUTHENTIC"}
          </div>
          <div className="inline-block mt-4 px-4 py-1.5 rounded-md bg-surface border border-border">
            <span className="text-base text-text-secondary">
              {getConfidenceScore(results)}% Confidence
            </span>
          </div>
        </div>
      </motion.div>

      {/* Collapsible API Response */}
      <motion.details
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-surface border border-border rounded-md overflow-hidden group"
      >
        <summary className="px-5 py-4 cursor-pointer select-none flex items-center justify-between text-sm font-medium text-text-secondary hover:text-text-primary transition-colors">
          <span className="flex items-center gap-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
            API Response
          </span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-transform group-open:rotate-180"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </summary>
        <div className="px-5 pb-4 border-t border-border-subtle pt-3">
          <div className="relative bg-surface border border-border rounded-md mt-2">
            <button
              onClick={() => {
                const json = JSON.stringify(
                  {
                    id: results.detection_id,
                    verdict: results.verdict,
                    confidence: results.confidence,
                    media_type: results.media_type_processed,
                    created_at: results.created_at,
                  },
                  null,
                  2,
                );
                navigator.clipboard.writeText(json);
                const btn = document.getElementById("copy-api-btn");
                if (btn) {
                  btn.textContent = "Copied!";
                  setTimeout(() => {
                    btn.textContent = "Copy";
                  }, 1500);
                }
              }}
              id="copy-api-btn"
              className="absolute top-2.5 right-2.5 px-2.5 py-1 text-xs font-medium text-accent border border-border rounded-md hover:bg-surface-raised hover:text-accent-hover transition-colors"
            >
              Copy
            </button>
            <pre className="font-mono text-sm text-text-secondary p-4 overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(
                {
                  id: results.detection_id,
                  verdict: results.verdict,
                  confidence: results.confidence,
                  media_type: results.media_type_processed,
                  created_at: results.created_at,
                },
                null,
                2,
              )}
            </pre>
          </div>
        </div>
      </motion.details>
    </div>
  );
}
