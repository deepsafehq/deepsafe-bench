"use client";

import React, { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AppNav } from "@/components/app-nav";
import { useAuth } from "@/components/auth-provider";
import { useAnalysis } from "@/hooks/useAnalysis";
import { UploadZone } from "@/components/upload-zone";
import { MediaPreview } from "@/components/media-preview";
import { AnalyzingState } from "@/components/analyzing-state";
import { AnalysisResultsPanel } from "@/components/analysis-results";
import { useApiHealth } from "@/hooks/useApiHealth";

export default function Home() {
  const { user, session, isLoading: authLoading } = useAuth();

  const {
    selectedFile,
    previewUrl,
    mediaType,
    fileName,
    fileSize,
    isAnalyzing,
    results,
    error,
    audioData,
    modelProgress,
    fileInputRef,
    startPolling,
    handleFileSelect,
    handleDemoMedia,
    handleAnalyze,
    resetSelection,
  } = useAnalysis(session?.access_token);

  const { apiOnline } = useApiHealth();

  // On mount, resume polling if there's an active job from a previous visit.
  useEffect(() => {
    if (authLoading || !user || !session?.access_token) return;
    const saved = sessionStorage.getItem("deepsafe_active_job");
    if (!saved) return;
    try {
      const job = JSON.parse(saved);
      if (job.jobId && job.startTime) {
        startPolling(job.jobId, session.access_token);
      }
    } catch {
      sessionStorage.removeItem("deepsafe_active_job");
    }
  }, [authLoading, user, session, startPolling]);

  const hasMedia =
    selectedFile !== null ||
    isAnalyzing ||
    results !== null ||
    previewUrl !== null;

  return (
    <div
      data-theme="app"
      className="min-h-screen bg-background text-text-primary flex flex-col"
    >
      <AppNav />

      {/* Main Content Split Screen */}
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden relative">
        {/* Left Column - Media Input */}
        <section className="flex-[0.9] p-6 md:p-10 flex flex-col items-center justify-center border-r border-border bg-surface relative z-0">
          <AnimatePresence mode="wait">
            {!hasMedia ? (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3 }}
              >
                <UploadZone
                  fileInputRef={fileInputRef}
                  onFileChange={handleFileSelect}
                  onDemoMedia={handleDemoMedia}
                  disabled={!apiOnline}
                />
              </motion.div>
            ) : (
              <motion.div
                key="preview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4 }}
                className="max-w-xl w-full flex flex-col items-center space-y-6"
              >
                {previewUrl && (
                  <MediaPreview
                    previewUrl={previewUrl}
                    mediaType={mediaType}
                    fileName={fileName}
                    fileSize={fileSize}
                    isAnalyzing={isAnalyzing}
                    audioData={audioData}
                    error={error}
                    onReset={resetSelection}
                    onAnalyze={handleAnalyze}
                    disabled={!apiOnline}
                  />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* Right Column - Results */}
        <section
          aria-live="polite"
          className="flex-[0.9] p-6 md:p-10 bg-background flex flex-col items-center justify-center overflow-y-auto relative z-0"
        >
          <AnimatePresence mode="wait">
            {isAnalyzing ? (
              <motion.div
                key="analyzing"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4 }}
              >
                <AnalyzingState modelProgress={modelProgress} />
              </motion.div>
            ) : !results ? (
              <motion.div
                key="awaiting"
                initial={{ opacity: 0, filter: "blur(4px)" }}
                animate={{ opacity: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, filter: "blur(4px)" }}
                transition={{ duration: 0.4 }}
                className="max-w-sm w-full space-y-5 text-center opacity-50"
              >
                <div className="w-20 h-20 rounded-md bg-surface flex items-center justify-center mx-auto mb-8 border border-border">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="32"
                    height="32"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-text-tertiary"
                  >
                    <rect width="18" height="18" x="3" y="3" rx="2" />
                    <path d="M3 9h18" />
                    <path d="M9 21V9" />
                  </svg>
                </div>
                <h3 className="text-2xl font-heading font-light text-text-secondary">
                  Ready to Analyze
                </h3>
                <p className="text-text-tertiary text-sm leading-relaxed">
                  Upload a file or try a sample to begin detection. Results will
                  appear here.
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{
                  staggerChildren: 0.15,
                  delayChildren: 0.1,
                  ease: "easeOut",
                }}
                className="w-full max-w-xl space-y-8"
              >
                <AnalysisResultsPanel results={results} />
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>
    </div>
  );
}
