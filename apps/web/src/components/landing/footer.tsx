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
                <a
                  href="#pricing"
                  className="hover:text-text-primary transition-colors"
                >
                  Pricing
                </a>
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
                  href="https://deepsafehq.github.io/deepsafe-bench/docs/endpoints"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-text-primary transition-colors"
                >
                  API Reference
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-medium text-text-primary mb-4">
              Company
            </h4>
            <ul className="space-y-2.5 text-sm text-text-secondary">
              <li>
                <a
                  href="mailto:contact@deepsafehq.github.io/deepsafe-bench"
                  className="hover:text-text-primary transition-colors"
                >
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="pt-8 border-t border-border-subtle flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-text-tertiary">
            &copy; {new Date().getFullYear()} DeepSafe AI, Inc. All rights
            reserved.
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
