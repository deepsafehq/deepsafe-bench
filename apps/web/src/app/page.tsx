"use client";

import React from "react";
import { LandingNavbar } from "@/components/landing/navbar";
import { Hero } from "@/components/landing/hero";
import { Metrics } from "@/components/landing/metrics";
import { Features } from "@/components/landing/features";
import { HowItWorks } from "@/components/landing/how-it-works";
import { ApiSection } from "@/components/landing/api-section";
import { BottomCta } from "@/components/landing/cta";
import { Footer } from "@/components/landing/footer";

export default function RootPage() {
  return <LandingPage />;
}

function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-text-primary font-sans selection:bg-accent/20 overflow-x-hidden">
      <LandingNavbar />
      <main>
        <Hero />
        <Metrics />
        <Features />
        <HowItWorks />
        <ApiSection />
        <BottomCta />
      </main>
      <Footer />
    </div>
  );
}
