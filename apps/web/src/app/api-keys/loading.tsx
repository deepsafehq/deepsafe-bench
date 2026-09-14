export default function Loading() {
  return (
    <div
      data-theme="app"
      role="status"
      aria-label="Loading"
      className="min-h-screen bg-background flex items-center justify-center"
    >
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
        <p className="text-sm text-text-secondary">Loading API keys...</p>
      </div>
    </div>
  );
}
