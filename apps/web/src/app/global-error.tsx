"use client";

import NextError from "next/error";

export default function GlobalError() {
  return (
    <html lang="en">
      <body>
        {/* `NextError` is the default Next.js error page component. Its type
        definition requires a `statusCode` prop. The App Router does not expose
        status codes for errors, so we pass 0 to render a generic message. */}
        <NextError statusCode={0} />
      </body>
    </html>
  );
}
