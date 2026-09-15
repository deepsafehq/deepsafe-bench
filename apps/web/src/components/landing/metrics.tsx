import React from "react";

// Every figure here comes from eval/results/predictions_medium.csv.gz and is
// reproducible with `deepsafe eval`. Nothing on this page is aspirational.
const stats = [
  { value: "24", label: "Models" },
  { value: "411", label: "Generators Tested" },
  { value: "15,499", label: "Samples Scored" },
  { value: "66.2%", label: "Fakes Caught" },
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
