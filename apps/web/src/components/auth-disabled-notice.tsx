"use client";

import Link from "next/link";

/**
 * Shown on account pages when the build has no auth configured.
 *
 * A self-hosted install runs without Supabase by default, which leaves sign-in,
 * the dashboard and key management with nothing to talk to. Saying so plainly
 * is better than rendering a form that throws on submit.
 */
export function AuthDisabledNotice({ page }: { page: string }) {
  return (
    <div className="min-h-screen bg-background text-text-primary flex items-center justify-center px-6">
      <div className="max-w-md border border-border-subtle p-8">
        <h1 className="text-xl font-medium mb-3">{page} is unavailable</h1>
        <p className="text-sm text-text-secondary leading-relaxed mb-4">
          This install has no authentication configured, so there are no
          accounts to sign in to. That is the default: on your own machine the
          API needs no credentials and you can call it directly.
        </p>
        <pre className="text-xs bg-card border border-border-subtle p-3 mb-4 overflow-x-auto">
          <code>{`curl -X POST http://localhost:8000/v1/detect \\
  -F "file=@photo.jpg"`}</code>
        </pre>
        <p className="text-sm text-text-secondary leading-relaxed mb-5">
          To enable accounts, set <code className="font-mono">SUPABASE_URL</code>
          , <code className="font-mono">NEXT_PUBLIC_SUPABASE_ANON_KEY</code> and{" "}
          <code className="font-mono">DEEPSAFE_REQUIRE_AUTH=true</code>. Do that
          before exposing the gateway to a network.
        </p>
        <Link href="/" className="text-sm text-accent hover:underline">
          Back to overview
        </Link>
      </div>
    </div>
  );
}
