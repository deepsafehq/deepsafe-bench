import React from "react";
import Link from "next/link";
import { DeepSafeLogo } from "@/components/logo";

export function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="max-w-[1200px] mx-auto px-6 pt-16 pb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          <div className="col-span-1">
            <div className="mb-4">
              <DeepSafeLogo size="sm" variant="light" />
            </div>
            <p className="text-sm text-text-tertiary leading-relaxed">
              Enterprise-grade deepfake detection infrastructure for modern
              applications.
            </p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-text-primary mb-4">
              Product
            </h4>
            <ul className="space-y-2.5 text-sm text-text-secondary">
              <li>
                <a
                  href="#features"
                  className="hover:text-text-primary transition-colors"
                >
                  Audio Detection
                </a>
              </li>
              <li>
                <a
                  href="#features"
                  className="hover:text-text-primary transition-colors"
                >
                  Video Forensics
                </a>
              </li>
              <li>
                <a
                  href="#features"
                  className="hover:text-text-primary transition-colors"
                >
                  Image Analysis
                </a>
              </li>
              <li>
                <Link
                  href="/benchmark"
                  className="hover:text-text-primary transition-colors"
                >
                  Benchmark
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-medium text-text-primary mb-4">
              Developers
            </h4>
            <ul className="space-y-2.5 text-sm text-text-secondary">
              <li>
                <a
                  href="https://deepsafehq.github.io/deepsafe-bench/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  Documentation
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/deepsafehq/deepsafe-bench/blob/main/REPRODUCING.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  Reproduce the results
                </a>
              </li>
              <li>
                <a
                  href="https://pypi.org/project/deepsafe-bench/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  PyPI package
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/deepsafe"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  Models &amp; datasets
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-medium text-text-primary mb-4">
              Project
            </h4>
            <ul className="space-y-2.5 text-sm text-text-secondary">
              <li>
                <a
                  href="https://github.com/deepsafehq/deepsafe-bench"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  Source code
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/deepsafehq/deepsafe-bench/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  Issues
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/deepsafehq/deepsafe-bench/blob/main/LICENSE"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  Licence
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="pt-8 border-t border-border-subtle flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-text-tertiary">
            &copy; {new Date().getFullYear()} DeepSafe. Free for non-commercial use
            under PolyForm Noncommercial 1.0.0.
          </p>
          <div className="flex gap-6 text-sm text-text-tertiary">
            <Link
              href="/privacy"
              className="hover:text-text-primary transition-colors"
            >
              Privacy
            </Link>
            <Link
              href="/terms"
              className="hover:text-text-primary transition-colors"
            >
              Terms
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
