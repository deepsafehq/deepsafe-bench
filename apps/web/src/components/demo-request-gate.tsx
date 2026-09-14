'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, ArrowRight, ChevronDown } from 'lucide-react';
import { DeepSafeLogo } from '@/components/logo';

const INDUSTRIES = [
  'Media & Journalism',
  'Government & Law Enforcement',
  'Corporate Security',
  'Financial Services',
  'Legal & Compliance',
  'Academia & Research',
  'Healthcare',
  'Technology',
  'Defense & Intelligence',
  'Insurance',
  'Other',
] as const;

const MEDIA_TYPES = [
  { value: 'image', label: 'Image Detection' },
  { value: 'audio', label: 'Audio / Voice Clone Detection' },
  { value: 'video', label: 'Video / Deepfake Detection' },
  { value: 'all', label: 'All Modalities' },
] as const;

const VOLUMES = [
  'Under 100 / month',
  '100 - 1,000 / month',
  '1,000 - 10,000 / month',
  '10,000+ / month',
  'Not sure yet',
] as const;

const inputClass =
  'w-full h-10 px-3 rounded-lg border border-border bg-background text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors';

const selectClass =
  'w-full h-10 px-3 rounded-lg border border-border bg-background text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors appearance-none cursor-pointer';

const labelClass = 'block text-xs font-medium text-text-secondary mb-1.5';

export function DemoRequestGate() {
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setSubmitting(true);
    const form = e.currentTarget;
    const data = new FormData(form);

    const res = await fetch('https://formspree.io/f/xzdkvyrj', {
      method: 'POST',
      body: data,
      headers: { Accept: 'application/json' },
    });

    setSubmitting(false);
    if (res.ok) {
      setSubmitted(true);
      form.reset();
    }
  }

  return (
    <div data-theme="app" className="min-h-screen bg-background flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.25, 1, 0.5, 1] }}
        className="w-full max-w-lg"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-6">
            <DeepSafeLogo size="md" showWordmark={false} variant="dark" />
            <span className="text-xl font-semibold tracking-tight text-text-primary">
              DeepSafe
            </span>
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-accent/10 text-accent text-xs font-medium mb-4">
            <Shield className="w-3 h-3" />
            Invite Only
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-text-primary mb-2">
            Request Demo Access
          </h1>
          <p className="text-sm text-text-secondary leading-relaxed max-w-md mx-auto">
            DeepSafe is currently in private preview. Tell us about your needs
            and we will reach out to set up a personalized demo.
          </p>
        </div>

        {/* Form Card */}
        <div className="rounded-xl border border-border bg-surface p-6">
          {submitted ? (
            <div className="text-center py-10">
              <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                  <path d="M20 6 9 17l-5-5"/>
                </svg>
              </div>
              <p className="text-base font-medium text-text-primary mb-1">
                Thank you for your interest
              </p>
              <p className="text-sm text-text-secondary max-w-xs mx-auto">
                We will review your request and reach out to schedule a
                personalized demo shortly.
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Name + Work Email */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="gate-name" className={labelClass}>
                    Full Name
                  </label>
                  <input
                    type="text"
                    id="gate-name"
                    name="name"
                    required
                    className={inputClass}
                    placeholder="Jane Smith"
                  />
                </div>
                <div>
                  <label htmlFor="gate-email" className={labelClass}>
                    Work Email
                  </label>
                  <input
                    type="email"
                    id="gate-email"
                    name="email"
                    required
                    className={inputClass}
                    placeholder="jane@company.com"
                  />
                </div>
              </div>

              {/* Organization + Role */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="gate-org" className={labelClass}>
                    Organization
                  </label>
                  <input
                    type="text"
                    id="gate-org"
                    name="organization"
                    required
                    className={inputClass}
                    placeholder="Company or institution"
                  />
                </div>
                <div>
                  <label htmlFor="gate-role" className={labelClass}>
                    Role / Title
                  </label>
                  <input
                    type="text"
                    id="gate-role"
                    name="role"
                    className={inputClass}
                    placeholder="e.g. Head of Security"
                  />
                </div>
              </div>

              {/* Industry */}
              <div>
                <label htmlFor="gate-industry" className={labelClass}>
                  Industry
                </label>
                <div className="relative">
                  <select
                    id="gate-industry"
                    name="industry"
                    required
                    defaultValue=""
                    className={selectClass}
                  >
                    <option value="" disabled>Select your industry</option>
                    {INDUSTRIES.map((ind) => (
                      <option key={ind} value={ind}>{ind}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary pointer-events-none" />
                </div>
              </div>

              {/* Detection Type */}
              <div>
                <label className={labelClass}>
                  What type of detection are you interested in?
                </label>
                <div className="grid grid-cols-2 gap-2 mt-1">
                  {MEDIA_TYPES.map(({ value, label }) => (
                    <label
                      key={value}
                      className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-border bg-background text-sm text-text-primary cursor-pointer hover:border-accent/40 transition-colors has-[:checked]:border-accent has-[:checked]:bg-accent/5"
                    >
                      <input
                        type="checkbox"
                        name="detection_type"
                        value={value}
                        className="accent-[var(--accent)] w-3.5 h-3.5"
                      />
                      <span className="text-xs">{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Expected Volume */}
              <div>
                <label htmlFor="gate-volume" className={labelClass}>
                  Expected Monthly Volume
                </label>
                <div className="relative">
                  <select
                    id="gate-volume"
                    name="expected_volume"
                    defaultValue=""
                    className={selectClass}
                  >
                    <option value="" disabled>How many files per month?</option>
                    {VOLUMES.map((vol) => (
                      <option key={vol} value={vol}>{vol}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary pointer-events-none" />
                </div>
              </div>

              {/* Use Case */}
              <div>
                <label htmlFor="gate-usecase" className={labelClass}>
                  Describe your use case
                </label>
                <textarea
                  id="gate-usecase"
                  name="use_case"
                  required
                  rows={3}
                  className="w-full px-3 py-2.5 rounded-lg border border-border bg-background text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors resize-none"
                  placeholder="What problem are you trying to solve? e.g. verifying user-submitted content, forensic investigation, content moderation..."
                />
              </div>

              {/* How did you hear */}
              <div>
                <label htmlFor="gate-referral" className={labelClass}>
                  How did you hear about DeepSafe?
                  <span className="text-text-tertiary font-normal"> (optional)</span>
                </label>
                <input
                  type="text"
                  id="gate-referral"
                  name="referral_source"
                  className={inputClass}
                  placeholder="e.g. Google, conference, colleague"
                />
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full h-11 flex items-center justify-center gap-2 text-sm font-medium bg-accent text-accent-foreground rounded-lg hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:pointer-events-none mt-2"
              >
                {submitting ? 'Sending...' : 'Request Demo Access'}
                {!submitting && <ArrowRight className="w-4 h-4" />}
              </button>
            </form>
          )}
        </div>

      </motion.div>
    </div>
  );
}
