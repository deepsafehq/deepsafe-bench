/**
 * Unmint Theme Configuration
 *
 * Customize your documentation's look and feel by modifying this file.
 * All colors, branding, and styling can be adjusted here.
 */

export const siteConfig = {
  // Site metadata
  name: "DeepSafe API",
  description:
    "Developer documentation for the DeepSafe deepfake detection API.",
  url: "https://deepsafehq.github.io/deepsafe-bench/docs",

  // Logo configuration
  logo: {
    src: "/logo.svg",
    alt: "DeepSafe",
    width: 40,
    height: 40,
  },

  // Navigation links
  links: {
    support: "mailto:contact@deepsafehq.github.io/deepsafe-bench",
  } as { support: string; github?: string; discord?: string },

  // Footer configuration
  footer: {
    copyright: "© 2026 DeepSafe AI, Inc. All rights reserved.",
    links: [
      { label: "Website", href: "https://deepsafehq.github.io/deepsafe-bench" },
      { label: "Dashboard", href: "http://localhost:3000" },
    ],
  },
};

export const themeConfig = {
  // Primary accent color - used for active states, links, highlights
  colors: {
    // Light mode
    light: {
      accent: "#047857",
      accentForeground: "#ffffff",
      accentMuted: "rgba(4, 120, 87, 0.1)",
    },
    dark: {
      accent: "#10B981",
      accentForeground: "#0f172a",
      accentMuted: "rgba(16, 185, 129, 0.1)",
    },
  },

  // Code block styling
  codeBlock: {
    light: {
      background: "#fafafa",
      titleBar: "#f3f4f6",
    },
    dark: {
      background: "#1a1a1f",
      titleBar: "#1f2937",
    },
  },

  // OG Image generation settings
  ogImage: {
    gradient: "linear-gradient(135deg, #ffffff 0%, #ecfdf5 50%, #6ee7b7 100%)",
    titleColor: "#0f172a",
    sectionColor: "#047857",
    logoUrl: "/logo.svg",
  },
};

// Export CSS variable values for use in Tailwind
export function getCSSVariables(mode: "light" | "dark") {
  const colors = themeConfig.colors[mode];
  return {
    "--accent": colors.accent,
    "--accent-foreground": colors.accentForeground,
    "--accent-muted": colors.accentMuted,
  };
}

/**
 * Get the site URL dynamically
 * Priority: NEXT_PUBLIC_SITE_URL > VERCEL_PROJECT_PRODUCTION_URL > VERCEL_URL > siteConfig.url
 * This allows OG images to work automatically on Vercel without configuration
 */
export function getSiteUrl(): string {
  if (process.env.NEXT_PUBLIC_SITE_URL) {
    return process.env.NEXT_PUBLIC_SITE_URL;
  }
  // Use production URL if available (custom domain)
  if (process.env.VERCEL_PROJECT_PRODUCTION_URL) {
    return `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`;
  }
  // Fallback to deployment URL for preview deployments
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  return siteConfig.url;
}
