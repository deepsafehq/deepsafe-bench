import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { DeepSafeLogo } from "@/components/logo";

export function PageNav() {
  return (
    <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-xl border-b border-border-subtle">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center">
          <DeepSafeLogo size="sm" variant="light" />
        </Link>
        <Link
          href="/"
          className="flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>
      </div>
    </nav>
  );
}
