"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";

interface AnalyzingStateProps {
  modelProgress: Record<string, Record<string, unknown>>;
}

export function AnalyzingState({ modelProgress }: AnalyzingStateProps) {
  const entries = Object.entries(modelProgress);
  const total = entries.length;
  const completed = entries.filter(
    ([, d]) => d.status === "complete" || d.status === "error",
  ).length;
  const running = entries.filter(([, d]) => d.status === "running").length;
  const progressPct = total > 0 ? (completed / total) * 100 : 0;

  const statusText =
    total === 0
      ? "Initializing detection pipeline"
      : `${completed} of ${total} models complete`;

  return (
    <AnimatePresence>
      <motion.div
        role="status"
        aria-label="Analyzing media"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -12 }}
        className="max-w-md w-full flex flex-col items-center justify-center space-y-10"
      >
        {/* Scanning animation */}
        <div className="relative w-32 h-32 flex items-center justify-center">
          {/* Outer pulsing ring */}
          <motion.div
            className="absolute inset-0 rounded-full border border-accent/20"
            animate={{ scale: [1, 1.15, 1], opacity: [0.3, 0.1, 0.3] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          />
          {/* Middle rotating ring */}
          <motion.div
            className="absolute inset-2 rounded-full border border-dashed border-accent/30"
            animate={{ rotate: 360 }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "linear",
            }}
          />
          {/* Inner ring with progress arc */}
          <svg className="absolute inset-4" viewBox="0 0 100 100" fill="none">
            {/* Track */}
            <circle
              cx="50"
              cy="50"
              r="46"
              stroke="var(--border)"
              strokeWidth="2"
            />
            {/* Progress arc */}
            <motion.circle
              cx="50"
              cy="50"
              r="46"
              stroke="var(--accent)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 46}`}
              strokeDashoffset={`${2 * Math.PI * 46 * (1 - progressPct / 100)}`}
              transform="rotate(-90 50 50)"
              style={{ filter: "drop-shadow(0 0 4px var(--accent))" }}
              initial={false}
              animate={{
                strokeDashoffset: 2 * Math.PI * 46 * (1 - progressPct / 100),
              }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
          </svg>
          {/* Center content */}
          <div className="relative z-10 flex flex-col items-center">
            {total > 0 ? (
              <span className="text-2xl font-heading font-medium text-text-primary tabular-nums">
                {Math.round(progressPct)}%
              </span>
            ) : (
              <motion.div
                className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  ease: "linear",
                }}
              />
            )}
          </div>
        </div>

        {/* Status text */}
        <div className="space-y-1.5 text-center">
          <h3 className="font-heading text-lg font-medium text-text-primary tracking-tight">
            Detection in Progress
          </h3>
          <p className="text-sm text-text-secondary">{statusText}</p>
          {running > 0 && (
            <p className="text-xs text-accent">
              {running} model{running > 1 ? "s" : ""} running
            </p>
          )}
        </div>

        {/* Model list */}
        {total > 0 && (
          <div className="w-full space-y-1.5">
            {entries.map(
              ([modelName, data]: [string, Record<string, unknown>], idx) => {
                const isComplete = data.status === "complete";
                const isError = data.status === "error";
                const isRunning = data.status === "running";
                const isFake = Boolean(data.is_fake);

                return (
                  <motion.div
                    key={modelName}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className={`flex items-center gap-3 rounded-md px-4 py-2.5 border transition-colors ${
                      isComplete
                        ? "bg-surface border-border"
                        : isRunning
                          ? "bg-accent-light border-accent/20"
                          : "bg-surface/50 border-border-subtle"
                    }`}
                  >
                    {/* Status indicator */}
                    <div className="flex-shrink-0">
                      {isComplete && !isFake && (
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 14 14"
                          fill="none"
                          className="text-accent"
                        >
                          <circle
                            cx="7"
                            cy="7"
                            r="6"
                            stroke="currentColor"
                            strokeWidth="1.5"
                          />
                          <path
                            d="M4.5 7L6.5 9L9.5 5"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      )}
                      {isComplete && isFake && (
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 14 14"
                          fill="none"
                          className="text-danger"
                        >
                          <circle
                            cx="7"
                            cy="7"
                            r="6"
                            stroke="currentColor"
                            strokeWidth="1.5"
                          />
                          <path
                            d="M5 5L9 9M9 5L5 9"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                          />
                        </svg>
                      )}
                      {isError && (
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 14 14"
                          fill="none"
                          className="text-warning"
                        >
                          <circle
                            cx="7"
                            cy="7"
                            r="6"
                            stroke="currentColor"
                            strokeWidth="1.5"
                          />
                          <path
                            d="M7 4.5V7.5M7 9.5V9"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                          />
                        </svg>
                      )}
                      {isRunning && (
                        <motion.div
                          className="w-3.5 h-3.5 border-[1.5px] border-accent border-t-transparent rounded-full"
                          animate={{ rotate: 360 }}
                          transition={{
                            duration: 0.8,
                            repeat: Infinity,
                            ease: "linear",
                          }}
                        />
                      )}
                      {data.status === "pending" && (
                        <div className="w-2 h-2 rounded-full bg-text-tertiary/40 ml-[3px]" />
                      )}
                    </div>

                    <span className="text-sm font-medium text-text-primary flex-1">
                      Analysis Module {idx + 1}
                    </span>

                    {isComplete && data.probability !== undefined && (
                      <span
                        className={`text-xs font-medium px-2 py-0.5 rounded-sm ${
                          isFake
                            ? "bg-danger-light text-danger"
                            : "bg-accent-light text-accent"
                        }`}
                      >
                        {(Number(data.probability) * 100).toFixed(1)}%
                      </span>
                    )}
                    {data.status === "pending" && (
                      <span className="text-xs text-text-tertiary">Queued</span>
                    )}
                    {isRunning && (
                      <span className="text-xs text-accent font-medium">
                        Running
                      </span>
                    )}
                    {isError && (
                      <span className="text-xs text-warning">Skipped</span>
                    )}
                  </motion.div>
                );
              },
            )}
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
