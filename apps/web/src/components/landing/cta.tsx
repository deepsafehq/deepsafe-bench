import React from "react";
import Link from "next/link";

export function BottomCta() {
  return (
    <section className="max-w-[720px] mx-auto px-6 py-24 sm:py-32 text-center">
      <h2 className="font-heading text-3xl md:text-[40px] font-light tracking-[-1px] leading-[1.15] text-text-primary mb-4">
        Run it against your own detector.
      </h2>
      <p className="text-base text-text-secondary mb-10 max-w-md mx-auto">
        Any detector implementing predict(path) &rarr; float can be scored across 411 generators. Adding a model is one file.
      </p>
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <Link
          href="https://github.com/deepsafehq/deepsafe-bench"
          className="h-12 px-8 flex items-center justify-center text-sm font-medium bg-accent text-accent-foreground rounded-full hover:bg-accent-hover transition-colors"
          style={{ transitionDuration: "var(--duration-fast)" }}
        >
          Get the code
        </Link>
        <a
          href="mailto:sales@deepsafehq.github.io/deepsafe-bench"
          className="text-sm text-text-secondary hover:text-text-primary transition-colors"
          style={{ transitionDuration: "var(--duration-fast)" }}
        >
          See the results
        </a>
      </div>
    </section>
  );
}
