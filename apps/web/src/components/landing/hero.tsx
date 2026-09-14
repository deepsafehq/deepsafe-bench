"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.08,
      duration: 0.35,
      ease: [0.25, 1, 0.5, 1] as [number, number, number, number],
    },
  }),
};

export function Hero() {
  return (
    <section className="max-w-[1200px] mx-auto px-6 pt-40 sm:pt-48 pb-20 sm:pb-32">
      <div className="flex flex-col items-center text-center max-w-[720px] mx-auto">
        <motion.p
          custom={0}
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          className="text-[13px] font-medium tracking-[1.5px] uppercase text-accent mb-6"
        >
          Enterprise Media Authentication
        </motion.p>

        <motion.h1
          custom={1}
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          className="font-heading text-4xl sm:text-5xl md:text-[64px] font-light tracking-[-2px] leading-[1.08] text-text-primary mb-6"
        >
          Verify what&apos;s real.
        </motion.h1>

        <motion.p
          custom={2}
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          className="text-lg text-text-secondary max-w-[560px] leading-relaxed mb-10"
        >
          DeepSafe&apos;s proprietary detection engine analyzes media with
          military-grade precision. Upload any image, video, or audio file and
          get a verdict in seconds.
        </motion.p>

        <motion.div
          custom={3}
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          className="flex flex-col sm:flex-row items-center gap-4 mb-4"
        >
          <Link
            href="http://localhost:3000/login"
            className="cta-animated h-12 px-8 flex items-center gap-2 text-sm font-medium text-white rounded-full"
          >
            Start Scanning
            <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="https://deepsafehq.github.io/deepsafe-bench/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-text-secondary hover:text-accent transition-colors flex items-center gap-1"
            style={{ transitionDuration: "var(--duration-fast)" }}
          >
            View Documentation
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
        </motion.div>

        <motion.p
          custom={4}
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          className="text-sm text-text-tertiary"
        >
          No credit card required. 200 free scans on sign up.
        </motion.p>
      </div>
    </section>
  );
}
