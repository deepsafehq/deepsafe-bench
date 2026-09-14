import React from "react";
import { ArrowRight } from "lucide-react";

const steps = [
  {
    number: "01",
    title: "Upload",
    description:
      "Drop any image, video, or audio file through the web dashboard. Or send it via our REST API.",
  },
  {
    number: "02",
    title: "Analyze",
    description:
      "Our proprietary detection engine examines the media across multiple forensic dimensions simultaneously.",
  },
  {
    number: "03",
    title: "Verdict",
    description:
      "Get a single confidence-scored verdict, authentic or synthetic, in seconds.",
  },
  {
    number: "04",
    title: "Integrate",
    description:
      "Embed detection into your own apps, moderation pipelines, or verification workflows via the API.",
  },
];

export function HowItWorks() {
  return (
    <section className="border-y border-border bg-surface">
      <div className="max-w-[1200px] mx-auto px-6 py-24 sm:py-32">
        <div className="text-center max-w-[560px] mx-auto mb-16">
          <p className="text-[13px] font-medium tracking-[1.5px] uppercase text-accent mb-4">
            How It Works
          </p>
          <h2 className="font-heading text-3xl md:text-[40px] font-light tracking-[-1px] leading-[1.15] text-text-primary">
            From upload to verdict in seconds.
          </h2>
        </div>

        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-12 md:gap-8">
          {steps.map((step) => (
            <div key={step.number} className="text-center md:text-left">
              <div className="font-heading text-5xl font-light text-accent/20 mb-4">
                {step.number}
              </div>
              <h3 className="font-heading text-xl font-normal tracking-[-0.5px] text-text-primary mb-3">
                {step.title}
              </h3>
              <p className="text-sm text-text-secondary leading-relaxed">
                {step.description}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <a
            href="https://deepsafehq.github.io/deepsafe-bench/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-hover transition-colors"
          >
            Read the API documentation
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </section>
  );
}
