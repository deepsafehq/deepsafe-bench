import React from "react";

const stats = [
  { value: "99.8%", label: "Accuracy" },
  { value: "<200ms", label: "Latency" },
  { value: "3", label: "Media Modalities" },
  { value: "24/7", label: "Always-On API" },
];

export function Metrics() {
  return (
    <section className="border-y border-border">
      <div className="max-w-[1200px] mx-auto px-6 py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:divide-x divide-border-subtle text-center">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="flex flex-col items-center justify-center space-y-2"
            >
              <div className="font-heading text-4xl md:text-5xl font-light tracking-tight text-text-primary">
                {stat.value}
              </div>
              <div className="text-[13px] font-medium tracking-[1.5px] uppercase text-accent">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
