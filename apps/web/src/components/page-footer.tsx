import Link from "next/link";

interface PageFooterProps {
  activePage: "privacy" | "terms";
}

export function PageFooter({ activePage }: PageFooterProps) {
  return (
    <footer className="border-t border-border py-8">
      <div className="max-w-4xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
        <p className="text-text-tertiary text-sm">
          &copy; {new Date().getFullYear()} DeepSafe AI, Inc. All rights
          reserved.
        </p>
        <div className="flex gap-6 text-sm text-text-tertiary">
          {activePage === "privacy" ? (
            <span className="text-text-primary">Privacy</span>
          ) : (
            <Link
              href="/privacy"
              className="hover:text-text-primary transition-colors"
            >
              Privacy
            </Link>
          )}
          {activePage === "terms" ? (
            <span className="text-text-primary">Terms</span>
          ) : (
            <Link
              href="/terms"
              className="hover:text-text-primary transition-colors"
            >
              Terms
            </Link>
          )}
        </div>
      </div>
    </footer>
  );
}
