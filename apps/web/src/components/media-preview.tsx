"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";

interface MediaPreviewProps {
  previewUrl: string;
  mediaType: string;
  fileName: string | null;
  fileSize: number | null;
  isAnalyzing: boolean;
  audioData: number[];
  error: string | null;
  onReset: () => void;
  onAnalyze: () => void;
  disabled?: boolean;
}

export function MediaPreview({
  previewUrl,
  mediaType,
  fileName,
  fileSize,
  isAnalyzing,
  audioData,
  error,
  onReset,
  onAnalyze,
  disabled,
}: MediaPreviewProps) {
  return (
    <div className="max-w-xl w-full flex flex-col items-center space-y-6">
      <div className="w-full aspect-video bg-background rounded-md border border-border overflow-hidden flex items-center justify-center relative p-2">
        <div className="w-full h-full rounded-sm overflow-hidden bg-background relative">
          {mediaType === "image" && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={previewUrl}
              alt="Preview"
              className="w-full h-full object-contain"
            />
          )}
          {mediaType === "video" && (
            <video
              src={previewUrl}
              controls
              className="w-full h-full object-contain"
            />
          )}
          {mediaType === "audio" && (
            <div className="w-full h-full flex flex-col items-center justify-center bg-background relative overflow-hidden">
              {/* Extracted Background Waveform */}
              <div className="absolute inset-0 flex items-center justify-center gap-1 opacity-40 pointer-events-none px-6">
                {(audioData.length > 0 ? audioData : Array(60).fill(10)).map(
                  (val, i) => {
                    const baseHeight = val;
                    const defaultOpacity = baseHeight / 100 + 0.1;
                    return (
                      <motion.div
                        key={i}
                        animate={{
                          height: isAnalyzing
                            ? [
                                `${baseHeight}%`,
                                `${Math.min(100, baseHeight * 1.6)}%`,
                                `${baseHeight}%`,
                              ]
                            : `${baseHeight}%`,
                          opacity: isAnalyzing
                            ? [defaultOpacity, 1, defaultOpacity]
                            : defaultOpacity,
                        }}
                        transition={{
                          repeat: isAnalyzing ? Infinity : 0,
                          duration: isAnalyzing ? 1.5 : 0,
                          delay: isAnalyzing ? i * 0.04 : 0,
                          ease: "easeInOut",
                        }}
                        className="flex-1 max-w-[6px] rounded-full bg-accent"
                        style={{
                          height: `${baseHeight}%`,
                          opacity: defaultOpacity,
                        }}
                      />
                    );
                  },
                )}
              </div>

              {/* Centered Pulse Ring (Only visible when analyzing) */}
              <AnimatePresence>
                {isAnalyzing && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="absolute w-40 h-40 rounded-full border-4 border-accent/20 animate-ping pointer-events-none"
                  />
                )}
              </AnimatePresence>

              <audio
                src={previewUrl}
                controls
                className="w-4/5 rounded-full relative z-10 bg-surface backdrop-blur-sm"
              />
            </div>
          )}
        </div>

        <button
          onClick={onReset}
          disabled={isAnalyzing}
          aria-label="Remove selected file"
          className="absolute top-4 right-4 w-9 h-9 bg-surface border border-border rounded-md flex items-center justify-center text-text-tertiary hover:text-text-primary hover:bg-surface-raised transition-all disabled:opacity-50 disabled:cursor-not-allowed z-10"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
          </svg>
        </button>
      </div>

      <div className="w-full space-y-2">
        <div className="w-full flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 bg-surface p-4 sm:p-5 rounded-md border border-border">
          <div className="flex flex-col min-w-0">
            <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-1">
              Selected File
            </span>
            <div className="truncate font-medium text-text-primary">
              {fileName || "File"}
            </div>
            <span className="text-xs text-text-secondary mt-0.5">
              {fileSize ? (fileSize / (1024 * 1024)).toFixed(2) : "\u2014"} MB
            </span>
          </div>
          <button
            onClick={onAnalyze}
            disabled={isAnalyzing || disabled}
            className={`w-full sm:w-auto px-6 py-3 rounded-md font-medium text-accent-foreground transition-all flex items-center justify-center gap-3 shrink-0 ${isAnalyzing || disabled ? "bg-accent/70 cursor-not-allowed opacity-50" : "bg-accent hover:bg-accent-hover"}`}
          >
            {isAnalyzing ? (
              <>
                <svg
                  className="animate-spin -ml-1 h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Analyzing...
              </>
            ) : (
              "Analyze Media"
            )}
          </button>
        </div>
        {mediaType === "audio" && !isAnalyzing && (
          <p className="text-xs text-text-tertiary px-1 flex items-center gap-1.5">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" x2="12" y1="8" y2="12" />
              <line x1="12" x2="12.01" y1="16" y2="16" />
            </svg>
            Audio analysis may take 2-3 minutes depending on file size.
          </p>
        )}
        {error && (
          <motion.div
            role="alert"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full p-4 bg-danger-light text-danger border border-danger/20 rounded-md text-sm flex items-center gap-3"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" x2="12" y1="8" y2="12" />
              <line x1="12" x2="12.01" y1="16" y2="16" />
            </svg>
            {error}
          </motion.div>
        )}
      </div>
    </div>
  );
}
