"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div
      data-theme="app"
      role="alert"
      className="min-h-screen bg-background flex items-center justify-center p-6"
    >
      <div className="max-w-md text-center space-y-4">
        <div className="text-red-400 text-5xl">!</div>
        <h2 className="text-xl font-semibold text-text-primary">
          Failed to load login
        </h2>
        <p className="text-sm text-text-secondary">
          {error.message || "An error occurred while loading the login page."}
        </p>
        <button
          onClick={reset}
          aria-label="Try again"
          className="px-4 py-2 bg-surface hover:bg-surface-raised text-text-primary rounded-md text-sm transition-colors"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
