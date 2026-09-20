"use client";

import { PageNav } from "@/components/page-nav";
import { PageFooter } from "@/components/page-footer";

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-background text-text-secondary">
      <PageNav />

      <main className="max-w-2xl mx-auto px-6 pt-32 pb-20">
        <h1 className="text-4xl font-heading font-light text-text-primary mb-8">
          Privacy
        </h1>

        <div className="space-y-6 text-[15px] leading-relaxed">
          <p>
            This site is a static project page. It has no backend, no
            analytics, no cookies and no accounts. Nothing you do here is
            recorded, because there is nothing here to record it.
          </p>

          <p>
            DeepSafe is software you run on your own hardware. Media you
            analyse is processed locally and never leaves your machine. There
            is no hosted service, so there is nothing for anyone to collect.
          </p>

          <h2 className="text-xl font-medium text-text-primary pt-4">
            If you self-host it
          </h2>
          <p>
            The gateway can optionally store detection history and issue API
            keys. That data lives in your database, under your control. Whether
            it is collected, retained or deleted is your decision, not ours. We
            never see it.
          </p>

          <h2 className="text-xl font-medium text-text-primary pt-4">
            Third parties
          </h2>
          <p>
            Model weights and datasets are downloaded from HuggingFace during
            setup, which is subject to{" "}
            <a
              href="https://huggingface.co/privacy"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              their privacy policy
            </a>
            . This page itself is hosted on GitHub Pages, which is subject to{" "}
            <a
              href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              GitHub&apos;s
            </a>
            .
          </p>

          <h2 className="text-xl font-medium text-text-primary pt-4">
            Questions
          </h2>
          <p>
            Email 
            <a
              href="mailto:deepsafe.hq@gmail.com"
              className="text-accent hover:underline"
            >
              deepsafe.hq@gmail.com
            </a>{" "}
            or open an issue on{" "}
            <a
              href="https://github.com/deepsafehq/deepsafe-bench/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              GitHub
            </a>
            .
          </p>
        </div>
      </main>

      <PageFooter activePage="privacy" />
    </div>
  );
}
