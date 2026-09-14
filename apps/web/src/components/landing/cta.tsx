import React from "react";
import Link from "next/link";

export function BottomCta() {
  return (
    <section className="max-w-[720px] mx-auto px-6 py-24 sm:py-32 text-center">
      <h2 className="font-heading text-3xl md:text-[40px] font-light tracking-[-1px] leading-[1.15] text-text-primary mb-4">
        Ready to verify your media?
      </h2>
      <p className="text-base text-text-secondary mb-10 max-w-md mx-auto">
        Join the organizations using DeepSafe to build trust and defend against
        AI-generated fraud.
      </p>
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <Link
          href="http://localhost:3000/login"
          className="h-12 px-8 flex items-center justify-center text-sm font-medium bg-accent text-accent-foreground rounded-full hover:bg-accent-hover transition-colors"
          style={{ transitionDuration: "var(--duration-fast)" }}
        >
          Get Started Free
        </Link>
        <a
          href="mailto:sales@deepsafehq.github.io/deepsafe-bench"
          className="text-sm text-text-secondary hover:text-text-primary transition-colors"
          style={{ transitionDuration: "var(--duration-fast)" }}
        >
          Contact Sales
        </a>
      </div>
    </section>
  );
}
