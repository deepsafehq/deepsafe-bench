"use client";

import { PageNav } from "@/components/page-nav";
import { PageFooter } from "@/components/page-footer";

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-background text-text-secondary">
      <PageNav />

      <main className="max-w-2xl mx-auto px-6 pt-32 pb-20">
        <h1 className="text-4xl font-heading font-light text-text-primary mb-8">
          Terms
        </h1>

        <div className="space-y-6 text-[15px] leading-relaxed">
          <p>
            There is no service to sign up for and nothing to buy. DeepSafe is
            software you download and run yourself, and the{" "}
            <a
              href="https://github.com/deepsafehq/deepsafe-bench/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              licence
            </a>{" "}
            is the agreement.
          </p>

          <h2 className="text-xl font-medium text-text-primary pt-4">
            Licence
          </h2>
          <p>
            PolyForm Noncommercial 1.0.0. Free for research, education,
            personal projects, charities, and government use. Commercial use is
            not permitted. This is a source-available licence, not an
            OSI-approved open source licence.
          </p>
          <p>
            That covers our code. Every third-party detection model keeps its
            own licence; see{" "}
            <a
              href="https://github.com/deepsafehq/deepsafe-bench/blob/main/THIRD_PARTY_NOTICES.md"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              THIRD_PARTY_NOTICES.md
            </a>
            . Evaluation datasets carry further restrictions from their
            original authors.
          </p>

          <h2 className="text-xl font-medium text-text-primary pt-4">
            No warranty
          </h2>
          <p>
            The software is provided as is, without warranty of any kind.
            Detection accuracy is limited and measured: the ensemble catches
            66.2% of fake media at a 6.5% false positive rate, and misses 93%
            of Sora video.{" "}
            <a
              href="https://github.com/deepsafehq/deepsafe-bench/blob/main/BENCHMARK.md"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              See the full results.
            </a>
          </p>

          <h2 className="text-xl font-medium text-text-primary pt-4">
            Do not use this as evidence
          </h2>
          <p>
            A detection score is an estimate, not proof. Do not use it to
            accuse anyone of anything, and do not rely on it for decisions that
            affect a person&apos;s rights, livelihood or reputation. Detection
            degrades sharply on generators absent from training, which in
            practice means the newest ones.
          </p>
        </div>
      </main>

      <PageFooter activePage="terms" />
    </div>
  );
}
