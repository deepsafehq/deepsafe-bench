import React from "react";
import { CheckCircle2, ArrowRight, Code2 } from "lucide-react";

const features = [
  "REST API with sync and async modes",
  "Flat per-scan pricing for any media type",
  "Confidence-scored verdicts for images, audio, and video",
  "200 free scans on sign up to get started",
];

export function ApiSection() {
  return (
    <section id="developers" className="border-t border-border">
      <div className="max-w-[1200px] mx-auto px-6 py-24 sm:py-32 grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
        <div>
          <div className="flex items-center gap-2 mb-6">
            <Code2 className="w-4 h-4 text-accent" />
            <p className="text-[13px] font-medium tracking-[1.5px] uppercase text-accent">
              Developer First
            </p>
          </div>
          <h2 className="font-heading text-3xl md:text-[40px] font-light tracking-[-1px] leading-[1.15] text-text-primary mb-6">
            Integrate in minutes.
          </h2>
          <p className="text-base text-text-secondary mb-8 leading-relaxed">
            Our REST API makes it simple to add deepfake detection to your
            moderation pipelines, identity verification flows, and content
            platforms.
          </p>

          <ul className="space-y-3 mb-8">
            {features.map((item) => (
              <li
                key={item}
                className="flex items-center gap-3 text-sm text-text-primary"
              >
                <CheckCircle2 className="w-4 h-4 text-accent shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <a
            href="https://deepsafehq.github.io/deepsafe-bench/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-hover transition-colors"
            style={{ transitionDuration: "var(--duration-fast)" }}
          >
            Read the documentation
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
        </div>

        <div className="rounded-md border border-border bg-[var(--dark-section-bg)] p-6 font-mono text-sm text-[var(--dark-section-text)] overflow-hidden">
          <div className="flex gap-1.5 mb-4">
            <div className="w-2.5 h-2.5 rounded-full bg-[#3B3B3B]" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#3B3B3B]" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#3B3B3B]" />
          </div>
          <pre className="overflow-x-auto text-[13px] leading-relaxed">
            <code>
              <span className="text-[#71717A]"># Analyze a video file</span>
              {"\n"}curl -X POST http://localhost:8000/v1/detect \{"\n"}
              {"  "}-H &quot;Authorization: Bearer ds_live_xyz...&quot; \{"\n"}
              {"  "}-F &quot;file=@suspicious_video.mp4&quot;{"\n"}
              {"\n"}
              <span className="text-[#71717A]"># Response</span>
              {"\n"}
              <span className="text-[#10B981]">{"{"}</span>
              {"\n"}
              {"  "}&quot;id&quot;: &quot;det_a1b2c3d4&quot;,{"\n"}
              {"  "}&quot;verdict&quot;:{" "}
              <span className="text-[#F59E0B]">&quot;fake&quot;</span>,{"\n"}
              {"  "}&quot;confidence&quot;: 0.94,{"\n"}
              {"  "}&quot;media_type&quot;: &quot;video&quot;,{"\n"}
              {"  "}&quot;created_at&quot;: &quot;2026-03-26T12:00:00Z&quot;
              {"\n"}
              <span className="text-[#10B981]">{"}"}</span>
            </code>
          </pre>
        </div>
      </div>
    </section>
  );
}
