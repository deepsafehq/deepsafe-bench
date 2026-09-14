"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { API_URL } from "@/lib/api";

export interface AnalysisResults {
  detection_id: string;
  verdict: string;
  confidence: number;
  is_likely_deepfake: boolean;
  deepfake_probability: number;
  media_type_processed: string;
  created_at: string;
}

const MAX_POLL_ATTEMPTS = 30;
const POLL_BASE_DELAY = 2000;
const POLL_MAX_DELAY = 16000;

export function useAnalysis(accessToken: string | null | undefined) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<string>("image");
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioData, setAudioData] = useState<number[]>([]);
  const [modelProgress, setModelProgress] = useState<
    Record<string, Record<string, unknown>>
  >({});

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollAttemptsRef = useRef(0);

  const extractAudioWaveform = useCallback(async (file: File) => {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new (
        window.AudioContext ||
        (window as unknown as Record<string, typeof AudioContext>)
          .webkitAudioContext
      )();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      const samples = 60;
      const blockSize = Math.floor(channelData.length / samples);
      const filteredData = [];
      for (let i = 0; i < samples; i++) {
        const blockStart = blockSize * i;
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
          sum += Math.abs(channelData[blockStart + j]);
        }
        filteredData.push(sum / blockSize);
      }
      const multiplier = Math.max(...filteredData);
      const normalizedData = filteredData.map((n) =>
        Math.max(5, (n / multiplier) * 100),
      );
      setAudioData(normalizedData);
    } catch (e) {
      console.error("Error extracting audio data", e);
      setAudioData(Array(60).fill(10));
    }
  }, []);

  const startPolling = useCallback((jobId: string, token: string) => {
    if (pollingRef.current) clearTimeout(pollingRef.current);
    pollAttemptsRef.current = 0;
    setIsAnalyzing(true);
    setResults(null);
    setError(null);

    const authHeader = { Authorization: `Bearer ${token}` };

    const scheduleNextPoll = (attempt: number) => {
      const delay = Math.min(
        POLL_BASE_DELAY * Math.pow(2, attempt),
        POLL_MAX_DELAY,
      );
      pollingRef.current = setTimeout(poll, delay);
    };

    const poll = async () => {
      const attempt = pollAttemptsRef.current;

      if (attempt >= MAX_POLL_ATTEMPTS) {
        sessionStorage.removeItem("deepsafe_active_job");
        setError("Analysis is taking too long. Please try again.");
        setIsAnalyzing(false);
        return;
      }

      try {
        const statusRes = await fetch(`${API_URL}/v1/results/${jobId}`, {
          headers: authHeader,
        });

        if (!statusRes.ok) {
          if (statusRes.status === 429) {
            pollAttemptsRef.current = attempt + 1;
            scheduleNextPoll(attempt + 1);
            return;
          }
          sessionStorage.removeItem("deepsafe_active_job");
          try {
            const errData = await statusRes.json();
            const detail = errData.detail;
            const msg =
              typeof detail === "string"
                ? detail
                : detail?.message || `Request failed (${statusRes.status})`;
            setError(msg);
          } catch {
            setError(`Request failed (${statusRes.status}). Please try again.`);
          }
          setIsAnalyzing(false);
          return;
        }

        const jobData = await statusRes.json();

        if (jobData.status === "complete") {
          sessionStorage.removeItem("deepsafe_active_job");
          setResults({
            detection_id: jobData.id,
            verdict: jobData.verdict,
            confidence: jobData.confidence,
            is_likely_deepfake: jobData.verdict === "fake",
            deepfake_probability:
              jobData.verdict === "fake"
                ? jobData.confidence
                : 1 - jobData.confidence,
            media_type_processed: jobData.media_type,
            created_at: jobData.created_at,
          });
          setIsAnalyzing(false);
        } else if (jobData.status === "failed") {
          sessionStorage.removeItem("deepsafe_active_job");
          setError(jobData.error || "Analysis failed");
          setIsAnalyzing(false);
        } else {
          pollAttemptsRef.current = attempt + 1;
          scheduleNextPoll(attempt);
        }
      } catch (pollErr: unknown) {
        sessionStorage.removeItem("deepsafe_active_job");
        const msg = pollErr instanceof Error ? pollErr.message : "";
        setError(
          msg.includes("NetworkError") || msg.includes("fetch")
            ? "Lost connection to the server. Please check your network and try again."
            : msg || "Something went wrong. Please try again.",
        );
        setIsAnalyzing(false);
      }
    };

    poll();
  }, []);

  const handleFileSelect = useCallback(
    (file: File) => {
      setSelectedFile(file);
      setFileName(file.name);
      setFileSize(file.size);
      setResults(null);
      setError(null);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      if (file.type.startsWith("video/")) {
        setMediaType("video");
      } else if (file.type.startsWith("audio/")) {
        setMediaType("audio");
        extractAudioWaveform(file);
      } else {
        setMediaType("image");
      }
    },
    [previewUrl, extractAudioWaveform],
  );

  const handleDemoMedia = useCallback(
    async (type: "image" | "video" | "audio") => {
      const extensions = { image: "jpg", video: "mp4", audio: "wav" };
      const mimeTypes = {
        image: "image/jpeg",
        video: "video/mp4",
        audio: "audio/wav",
      };
      const demoPath = `/samples/sample_${type}.${extensions[type]}`;

      try {
        const res = await fetch(demoPath);
        const blob = await res.blob();
        const file = new File([blob], `sample_${type}.${extensions[type]}`, {
          type: mimeTypes[type],
        });
        setSelectedFile(file);
        setFileName(file.name);
        setFileSize(file.size);
        setPreviewUrl(demoPath);
        setMediaType(type);
        setResults(null);
        setError(null);
        if (type === "audio") {
          extractAudioWaveform(file);
        }
      } catch {
        setError(`Failed to load demo ${type}`);
      }
    },
    [extractAudioWaveform],
  );

  const resetSelection = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setFileName(null);
    setFileSize(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    sessionStorage.removeItem("deepsafe_active_job");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [previewUrl]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    if (!accessToken) {
      setError("Session expired. Please sign in again.");
      setIsAnalyzing(false);
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    setResults(null);
    setModelProgress({});

    const authHeader = { Authorization: `Bearer ${accessToken}` };

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const uploadStart = Date.now();

      const response = await fetch(`${API_URL}/v1/detect`, {
        method: "POST",
        headers: authHeader,
        body: formData,
      });

      if (response.status === 402) {
        setError(
          "You've used all your free scans. Upgrade at deepsafehq.github.io/deepsafe-bench/#pricing for more.",
        );
        setIsAnalyzing(false);
        return;
      }

      if (response.status === 429) {
        const errData = await response.json().catch(() => null);
        const retryAfter = errData?.detail?.retry_after;
        const mins = retryAfter ? Math.ceil(retryAfter / 60) : null;
        setError(
          mins && mins >= 60
            ? "You've reached the daily scan limit (10/day on the free plan). Try again tomorrow."
            : `Too many requests. Please wait ${mins || "a few"} minutes and try again.`,
        );
        setIsAnalyzing(false);
        return;
      }

      if (!response.ok) {
        const errorData = await response.json();
        const detail = errorData.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : detail?.message || "Analysis failed. Please try again.";
        throw new Error(msg);
      }

      const data = await response.json();

      if (response.status === 200 && data.verdict) {
        setResults({
          detection_id: data.id,
          verdict: data.verdict,
          confidence: data.confidence,
          is_likely_deepfake: data.verdict === "fake",
          deepfake_probability:
            data.verdict === "fake" ? data.confidence : 1 - data.confidence,
          media_type_processed: data.media_type,
          created_at: data.created_at,
        });
        setIsAnalyzing(false);
        return;
      }

      const jobId = data.id;
      sessionStorage.setItem(
        "deepsafe_active_job",
        JSON.stringify({
          jobId,
          startTime: uploadStart,
          previewUrl,
          mediaType,
          fileName,
          fileSize,
        }),
      );
      startPolling(jobId, accessToken);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "";
      if (msg.includes("NetworkError") || msg.includes("fetch")) {
        setError(
          "Could not reach the server. Please check your network and try again.",
        );
      } else {
        setError(msg || "Something went wrong. Please try again.");
      }
      setIsAnalyzing(false);
    }
  }, [
    selectedFile,
    accessToken,
    previewUrl,
    mediaType,
    fileName,
    fileSize,
    startPolling,
  ]);

  // Clean up polling timeout on unmount.
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearTimeout(pollingRef.current);
    };
  }, []);

  return {
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
  };
}
