"use client";

import React from "react";
import { Film, AudioLines, ImageIcon } from "lucide-react";

function DotGrid() {
  return (
    <div
      className="absolute inset-0 opacity-40"
      style={{
        backgroundImage: "radial-gradient(var(--border) 1px, transparent 1px)",
        backgroundSize: "16px 16px",
      }}
    />
  );
}

function VideoIllustration() {
  const frames = [0.6, 0.75, 0.9, 1, 0.9, 0.75, 0.6];
  return (
    <div className="w-full aspect-[4/3] rounded-md border border-border bg-surface overflow-hidden relative flex flex-col items-center justify-center gap-4 p-6 sm:p-8">
      <DotGrid />
      <div className="relative z-10 flex items-end gap-2">
        {frames.map((opacity, i) => {
          const isActive = i === 3;
          return (
            <div
              key={i}
              className={`relative rounded-sm border transition-all duration-300 ${
                isActive
                  ? "w-16 h-12 sm:w-20 sm:h-14 border-accent shadow-md -translate-y-1"
                  : "w-12 h-9 sm:w-14 sm:h-10 border-border"
              }`}
              style={{ opacity }}
            >
              <div className="absolute inset-0 flex flex-col gap-1.5 p-2">
                <div
                  className={`h-[2px] rounded-full ${isActive ? "bg-accent/30" : "bg-border"} w-3/4`}
                />
                <div
                  className={`h-[2px] rounded-full ${isActive ? "bg-accent/20" : "bg-border-subtle"} w-1/2`}
                />
              </div>
              {isActive && (
                <div className="absolute inset-0 overflow-hidden rounded-sm">
                  <div
                    className="absolute left-0 right-0 h-[1px] bg-accent/50"
                    style={{
                      animation: "scanLine 2.5s ease-in-out infinite",
                    }}
                  />
                </div>
              )}
              {(i === 1 || i === 5) && (
                <div className="absolute -top-1 -right-1 w-1.5 h-1.5 rounded-full bg-danger" />
              )}
            </div>
          );
        })}
      </div>
      <div className="relative z-10 w-full max-w-[240px] sm:max-w-[280px]">
        <div className="w-full h-[2px] bg-border relative">
          {Array.from({ length: 9 }).map((_, i) => (
            <div
              key={i}
              className="absolute top-0 w-[1px] h-1.5 bg-border"
              style={{ left: `${(i / 8) * 100}%` }}
            />
          ))}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-accent"
            style={{ left: "50%" }}
          />
        </div>
        <div className="flex justify-between mt-1.5">
          <span className="text-[9px] font-mono text-text-tertiary">00:00</span>
          <span className="text-[9px] font-mono text-text-tertiary">00:14</span>
        </div>
      </div>
      <style jsx>{`
        @keyframes scanLine {
          0%,
          100% {
            top: 10%;
          }
          50% {
            top: 85%;
          }
        }
      `}</style>
    </div>
  );
}

function AudioIllustration() {
  const barHeights = [
    12, 18, 14, 24, 32, 28, 38, 44, 48, 42, 46, 44, 40, 36, 38, 34, 28, 32, 24,
    18, 22, 16, 14, 10,
  ];
  const anomalyStart = 7;
  const anomalyEnd = 14;

  return (
    <div className="w-full aspect-[4/3] rounded-md border border-border bg-surface overflow-hidden relative flex flex-col items-center justify-center gap-3 p-6 sm:p-8">
      <DotGrid />
      <svg
        className="relative z-10 w-full max-w-[280px] h-8"
        viewBox="0 0 280 32"
        fill="none"
        preserveAspectRatio="none"
      >
        <path
          d="M0,16 C20,8 40,22 70,12 C100,2 120,20 140,10 C160,0 180,18 210,14 C240,10 260,20 280,16"
          stroke="var(--border)"
          strokeWidth="1.5"
          fill="none"
        />
        <path
          d="M0,16 C20,8 40,22 70,12 C100,2 120,20 140,10 C160,0 180,18 210,14 C240,10 260,20 280,16"
          stroke="var(--accent)"
          strokeWidth="1.5"
          fill="none"
          strokeDasharray="280"
          strokeDashoffset="100"
          style={{ clipPath: "inset(0 25% 0 30%)" }}
        />
      </svg>
      <div className="relative z-10">
        <div className="flex items-end gap-[3px] h-[52px]">
          {barHeights.map((h, i) => {
            const isAnomaly = i >= anomalyStart && i <= anomalyEnd;
            const isEdge = i === anomalyStart || i === anomalyEnd;
            return (
              <div
                key={i}
                className={`w-[3px] rounded-full ${
                  isAnomaly
                    ? isEdge
                      ? "bg-accent/50"
                      : "bg-accent"
                    : "bg-border"
                }`}
                style={{
                  height: `${h}px`,
                  animation: `barPulse 2s ease-in-out infinite`,
                  animationDelay: `${i * 0.08}s`,
                }}
              />
            );
          })}
        </div>
        {/* Anomaly label */}
        <div className="absolute -top-4 left-1/2 -translate-x-1/2">
          <span className="text-[9px] font-mono text-accent bg-[var(--background)]/80 px-1 rounded-sm">
            Anomaly detected
          </span>
        </div>
      </div>
      <div className="relative z-10 w-full max-w-[240px]">
        <div className="w-full h-[1px] bg-border" />
        <div className="flex justify-between mt-1">
          <span className="text-[9px] font-mono text-text-tertiary">0 Hz</span>
          <span className="text-[9px] font-mono text-text-tertiary">8 kHz</span>
          <span className="text-[9px] font-mono text-text-tertiary">
            16 kHz
          </span>
        </div>
      </div>
      <style jsx>{`
        @keyframes barPulse {
          0%,
          100% {
            transform: scaleY(1);
          }
          50% {
            transform: scaleY(0.88);
          }
        }
      `}</style>
    </div>
  );
}

function ImageIllustration() {
  const gridSize = { cols: 12, rows: 9 };
  const anomalyCells = new Set([
    "7-5",
    "8-5",
    "9-5",
    "10-5",
    "7-6",
    "8-6",
    "9-6",
    "10-6",
    "7-7",
    "8-7",
    "9-7",
    "10-7",
    "8-8",
    "9-8",
  ]);
  const anomalyCore = new Set(["8-6", "9-6", "8-7", "9-7"]);

  function getCellColor(col: number, row: number): string {
    const key = `${col}-${row}`;
    if (anomalyCore.has(key)) return "bg-accent";
    if (anomalyCells.has(key)) return "bg-accent/40";
    const shade = ((col + row) * 7 + col * 3) % 4;
    const shades = [
      "bg-surface",
      "bg-border-subtle/50",
      "bg-border/30",
      "bg-surface",
    ];
    return shades[shade];
  }

  return (
    <div className="w-full aspect-[4/3] rounded-md border border-border bg-surface overflow-hidden relative flex items-center justify-center p-6 sm:p-8">
      <DotGrid />
      <div className="relative z-10">
        <div
          className="grid gap-[2px]"
          style={{ gridTemplateColumns: `repeat(${gridSize.cols}, 1fr)` }}
        >
          {Array.from({ length: gridSize.rows }).map((_, row) =>
            Array.from({ length: gridSize.cols }).map((_, col) => (
              <div
                key={`${col}-${row}`}
                className={`w-3 h-3 sm:w-3.5 sm:h-3.5 rounded-[1px] ${getCellColor(col, row)}`}
                style={
                  anomalyCells.has(`${col}-${row}`)
                    ? {
                        animation: "cellPulse 3s ease-in-out infinite",
                        animationDelay: `${(col + row) * 0.1}s`,
                      }
                    : undefined
                }
              />
            )),
          )}
        </div>
        {/* Bounding box */}
        <div
          className="absolute border border-dashed border-accent/60 rounded-sm pointer-events-none"
          style={{
            top: `calc(${5 * (100 / gridSize.rows)}% - 4px)`,
            left: `calc(${7 * (100 / gridSize.cols)}% - 4px)`,
            width: `calc(${4 * (100 / gridSize.cols)}% + 8px)`,
            height: `calc(${4 * (100 / gridSize.rows)}% + 8px)`,
          }}
        >
          <span className="absolute -top-4 right-0 text-[9px] font-mono text-accent bg-[var(--background)]/80 px-1 rounded-sm">
            87.4%
          </span>
        </div>
        {/* Magnifier */}
        <div className="absolute -top-2 -right-4 sm:-right-6 w-14 h-14 sm:w-16 sm:h-16 rounded-full border-2 border-accent/40 bg-[var(--background)] shadow-lg overflow-hidden flex items-center justify-center">
          <div className="grid grid-cols-3 gap-[2px]">
            {[
              "bg-accent/40",
              "bg-accent",
              "bg-accent/40",
              "bg-accent",
              "bg-accent",
              "bg-accent/60",
              "bg-accent/40",
              "bg-accent/60",
              "bg-accent/40",
            ].map((color, i) => (
              <div
                key={i}
                className={`w-3 h-3 sm:w-3.5 sm:h-3.5 rounded-[1px] ${color}`}
              />
            ))}
          </div>
        </div>
      </div>
      <style jsx>{`
        @keyframes cellPulse {
          0%,
          100% {
            opacity: 0.75;
          }
          50% {
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

const features = [
  {
    overline: "Video Forensics",
    title: "Every frame tells the truth.",
    description:
      "Upload any video and get frame-by-frame forensic analysis. DeepSafe detects facial manipulation, lip-sync inconsistencies, and temporal artifacts invisible to the naked eye.",
    icon: Film,
    illustration: VideoIllustration,
  },
  {
    overline: "Audio Analysis",
    title: "Hear what algorithms hide.",
    description:
      "Voice clones and synthetic speech leave subtle spectral fingerprints. DeepSafe catches frequency anomalies and phase artifacts across speech and music.",
    icon: AudioLines,
    illustration: AudioIllustration,
  },
  {
    overline: "Image Detection",
    title: "See beyond the pixel.",
    description:
      "From GANs to diffusion models, every generation method leaves traces. DeepSafe identifies manipulation artifacts at the sub-pixel level across all major AI image generators.",
    icon: ImageIcon,
    illustration: ImageIllustration,
  },
];

export function Features() {
  return (
    <section
      id="features"
      className="max-w-[1200px] mx-auto px-6 py-24 sm:py-32"
    >
      <div className="text-center max-w-[560px] mx-auto mb-20">
        <p className="text-[13px] font-medium tracking-[1.5px] uppercase text-accent mb-4">
          Detection Capabilities
        </p>
        <h2 className="font-heading text-3xl md:text-[40px] font-light tracking-[-1px] leading-[1.15] text-text-primary">
          Comprehensive defense across every modality.
        </h2>
      </div>

      <div className="space-y-24 sm:space-y-32">
        {features.map((feature, i) => {
          const Icon = feature.icon;
          const Illustration = feature.illustration;
          const reversed = i % 2 !== 0;
          return (
            <div
              key={feature.overline}
              className={`flex flex-col ${reversed ? "md:flex-row-reverse" : "md:flex-row"} gap-12 md:gap-16 items-center`}
            >
              <div className="flex-1 max-w-md">
                <div className="flex items-center gap-2 mb-4">
                  <Icon className="w-4 h-4 text-accent" />
                  <p className="text-[13px] font-medium tracking-[1.5px] uppercase text-accent">
                    {feature.overline}
                  </p>
                </div>
                <h3 className="font-heading text-2xl md:text-[28px] font-light tracking-[-0.5px] leading-[1.2] text-text-primary mb-4">
                  {feature.title}
                </h3>
                <p className="text-base text-text-secondary leading-relaxed">
                  {feature.description}
                </p>
              </div>
              <div className="flex-1 w-full">
                <Illustration />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
