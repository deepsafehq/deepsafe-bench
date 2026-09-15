"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Menu, X } from "lucide-react";
import { DeepSafeLogo } from "@/components/logo";

const NAV_ITEMS = [
  { label: "Benchmark", href: "/benchmark" },
  { label: "Features", href: "#features" },
  { label: "Developers", href: "#developers" },
];

export function LandingNavbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-[border-color,backdrop-filter] ${
        scrolled
          ? "border-b border-border-subtle backdrop-blur-sm bg-[var(--background)]/80"
          : "border-b border-transparent"
      }`}
      style={{
        transitionDuration: "var(--duration-normal)",
        transitionTimingFunction: "var(--ease-subtle)",
      }}
    >
      <div className="max-w-[1200px] mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center">
          <DeepSafeLogo size="md" variant="light" />
        </Link>

        <nav className="hidden md:flex items-center gap-8 text-sm text-text-secondary">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="hover:text-text-primary transition-colors"
              style={{ transitionDuration: "var(--duration-fast)" }}
            >
              {item.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <Link
            href="https://github.com/deepsafehq/deepsafe-bench"
            className="hidden md:block text-sm text-text-secondary hover:text-text-primary transition-colors"
            style={{ transitionDuration: "var(--duration-fast)" }}
          >
            Sign In
          </Link>
          <Link
            href="https://github.com/deepsafehq/deepsafe-bench"
            className="hidden sm:flex h-9 px-5 items-center text-sm font-medium bg-accent text-accent-foreground rounded-full hover:bg-accent-hover transition-colors"
            style={{ transitionDuration: "var(--duration-fast)" }}
          >
            Get Started
          </Link>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-text-secondary hover:text-text-primary transition-colors"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {mobileMenuOpen && (
        <div className="md:hidden border-t border-border-subtle bg-[var(--background)]">
          <nav className="flex flex-col px-6 py-4 gap-1">
            {NAV_ITEMS.map((item) => (
              <a
                key={item.label}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="text-sm text-text-secondary hover:text-text-primary py-2"
              >
                {item.label}
              </a>
            ))}
            <div className="border-t border-border-subtle pt-3 mt-2 flex flex-col gap-2">
              <Link
                href="https://github.com/deepsafehq/deepsafe-bench"
                className="w-full py-2.5 text-sm font-medium border border-border text-text-secondary hover:text-text-primary text-center rounded-md transition-colors"
              >
                Sign In
              </Link>
              <Link
                href="https://github.com/deepsafehq/deepsafe-bench"
                className="w-full py-2.5 text-sm font-medium bg-accent text-accent-foreground text-center rounded-full"
              >
                Get Started
              </Link>
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
