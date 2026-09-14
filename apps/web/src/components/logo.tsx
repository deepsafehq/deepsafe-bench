"use client";

import React from "react";

interface DeepSafeLogoProps {
  size?: "sm" | "md" | "lg";
  showWordmark?: boolean;
  variant?: "light" | "dark";
}

export function DeepSafeLogo({
  size = "md",
  showWordmark = true,
  variant = "light",
}: DeepSafeLogoProps) {
  const dimensions = {
    sm: { w: 20, h: 22.5, text: "text-base" },
    md: { w: 24, h: 27, text: "text-lg" },
    lg: { w: 64, h: 72, text: "text-3xl" },
  };

  const { w, h, text } = dimensions[size];
  const shieldColor = variant === "light" ? "#047857" : "#10B981";
  const wordmarkColor =
    variant === "light" ? "text-[#1A1A1A]" : "text-[#E4E4E7]";

  return (
    <span className="inline-flex items-center gap-2">
      <svg
        width={w}
        height={h}
        viewBox="0 0 28 31.5"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <path
          d="M14 0L27 7v10c0 8-5.5 12.5-13 14.5C6.5 29.5 1 25 1 17V7l13-7z"
          fill={shieldColor}
          fillOpacity="0.15"
          stroke={shieldColor}
          strokeWidth="1.5"
        />
        <path
          d="M14 6L23 11v7c0 5.5-3.8 8.6-9 10-5.2-1.4-9-4.5-9-10v-7l9-5z"
          fill={shieldColor}
          fillOpacity="0.08"
          stroke={shieldColor}
          strokeWidth="1"
          strokeOpacity="0.5"
        />
        <circle cx="14" cy="16" r="3" fill={shieldColor} fillOpacity="0.6" />
        <line
          x1="14"
          y1="8"
          x2="14"
          y2="24"
          stroke={shieldColor}
          strokeWidth="0.5"
          strokeOpacity="0.3"
        />
        <line
          x1="6"
          y1="16"
          x2="22"
          y2="16"
          stroke={shieldColor}
          strokeWidth="0.5"
          strokeOpacity="0.3"
        />
      </svg>
      {showWordmark && (
        <span
          className={`${text} font-heading font-light tracking-tight ${wordmarkColor}`}
        >
          DeepSafe
        </span>
      )}
    </span>
  );
}
