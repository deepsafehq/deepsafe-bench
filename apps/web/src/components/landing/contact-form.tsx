'use client';

import React, { useState } from 'react';

function ContactForm() {
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
    <section id="contact" className="py-24 px-6 border-t border-border">
      <div className="max-w-lg mx-auto">
        <div className="text-center mb-10">
          <h2 className="font-heading text-3xl md:text-[40px] font-light tracking-[-1px] leading-[1.15] text-text-primary mb-4">
            Request a Demo
          </h2>
          <p className="text-base text-text-secondary max-w-md mx-auto">
            See DeepSafe in action with your own media. We will set up a
            personalized demo session for your team.
          </p>
        </div>

        {submitted ? (
          <div className="text-center py-12 px-6 rounded-xl border border-accent/20 bg-accent/5">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 text-accent">
              <path d="M20 6 9 17l-5-5"/>
            </svg>
            <p className="text-lg font-medium text-text-primary mb-1">
              Thank you for your interest
            </p>
            <p className="text-sm text-text-secondary">
              We will reach out to schedule your demo shortly.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-text-secondary mb-1.5">
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  required
                  className="w-full h-11 px-3.5 rounded-lg border border-border bg-background text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors"
                  placeholder="Your name"
                />
              </div>
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-text-secondary mb-1.5">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  required
                  className="w-full h-11 px-3.5 rounded-lg border border-border bg-background text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors"
                  placeholder="you@company.com"
                />
              </div>
            </div>
            <div>
              <label htmlFor="company" className="block text-sm font-medium text-text-secondary mb-1.5">
                Company
                <span className="text-text-tertiary font-normal"> (optional)</span>
              </label>
              <input
                type="text"
                id="company"
                name="company"
                className="w-full h-11 px-3.5 rounded-lg border border-border bg-background text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors"
                placeholder="Your organization"
              />
            </div>
            <div>
              <label htmlFor="message" className="block text-sm font-medium text-text-secondary mb-1.5">
                Message
              </label>
              <textarea
                id="message"
                name="message"
                required
                rows={4}
                className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-background text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors resize-none"
                placeholder="Tell us about your use case"
              />
            </div>
            <button
              type="submit"
              disabled={submitting}
              className="w-full h-12 flex items-center justify-center text-sm font-medium bg-accent text-accent-foreground rounded-full hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:pointer-events-none"
              style={{ transitionDuration: 'var(--duration-fast)' }}
            >
              {submitting ? 'Sending...' : 'Request a Demo'}
            </button>
          </form>
        )}
      </div>
    </section>
  );
}

export { ContactForm };
