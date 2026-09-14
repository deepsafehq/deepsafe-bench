"use client";

import React from "react";

interface UploadZoneProps {
  fileInputRef: React.RefObject<HTMLInputElement>;
  onFileChange: (file: File) => void;
  onDemoMedia: (type: "image" | "video" | "audio") => void;
  disabled?: boolean;
}

export function UploadZone({
  fileInputRef,
  onFileChange,
  onDemoMedia,
  disabled,
}: UploadZoneProps) {
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileChange(file);
    }
  };

  return (
    <div className={`max-w-md w-full space-y-8 text-center${disabled ? ' pointer-events-none opacity-50' : ''}`}>
      <div className="space-y-2">
        <h2 className="font-heading text-4xl font-light tracking-tight text-text-primary">
          Analyze Media
        </h2>
        <p className="text-text-secondary leading-relaxed text-sm">
          Upload an image, audio, or video file to detect AI manipulation and
          deepfakes using our proprietary detection technology.
        </p>
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload media file"
        className="group border-2 border-dashed border-border rounded-md p-12 hover:border-accent hover:bg-accent-light transition-all duration-200 cursor-pointer flex flex-col items-center justify-center gap-5"
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleInputChange}
          className="hidden"
          accept="image/*,video/*,audio/*"
          aria-label="Select media file for analysis"
        />
        <div className="w-16 h-16 rounded-full bg-surface-raised flex items-center justify-center text-text-tertiary group-hover:text-accent transition-colors duration-200">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" x2="12" y1="3" y2="15" />
          </svg>
        </div>
        <div>
          <div className="font-medium text-lg text-text-primary">
            Click to upload
          </div>
          <div className="text-sm text-text-tertiary mt-1">
            MP4, WAV, JPG, PNG (Max 50MB)
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 sm:gap-3 pt-2 w-full max-w-sm mx-auto">
        {(["image", "audio", "video"] as const).map((type) => (
          <button
            key={type}
            onClick={() => onDemoMedia(type)}
            className="py-2.5 text-xs font-medium rounded-full border border-border text-text-secondary hover:text-text-primary hover:border-accent transition-all whitespace-nowrap"
          >
            Sample {type.charAt(0).toUpperCase() + type.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
}
