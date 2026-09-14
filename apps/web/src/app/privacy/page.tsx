"use client";

import { PageNav } from "@/components/page-nav";
import { PageFooter } from "@/components/page-footer";

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-background text-text-secondary">
      <PageNav />

      {/* Content */}
      <main className="max-w-4xl mx-auto px-6 pt-32 pb-20">
        <h1 className="text-4xl font-heading font-light text-text-primary mb-2">
          Privacy Policy
        </h1>
        <p className="text-text-tertiary mb-12">Last updated: March 26, 2026</p>

        <div className="space-y-10 text-[15px] leading-relaxed">
          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              1. Introduction
            </h2>
            <p>
              DeepSafe AI, Inc. (&quot;DeepSafe,&quot; &quot;we,&quot;
              &quot;us,&quot; or &quot;our&quot;) operates the DeepSafe platform
              at deepsafehq.github.io/deepsafe-bench, including the web application, REST API, and
              related services (collectively, the &quot;Service&quot;). This
              Privacy Policy explains how we collect, use, store, and protect
              your information when you use our Service.
            </p>
            <p className="mt-3">
              By using DeepSafe, you agree to the collection and use of
              information in accordance with this policy. If you do not agree
              with this policy, please do not use the Service.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              2. Information We Collect
            </h2>
            <h3 className="text-lg font-heading font-light text-text-primary mt-6 mb-3">
              2.1 Account Information
            </h3>
            <p>
              When you create an account, we collect your email address, display
              name, and authentication credentials (managed via our
              authentication provider). If you sign in via a social login
              provider (e.g., Google, GitHub), we receive the profile
              information you authorize.
            </p>

            <h3 className="text-lg font-heading font-light text-text-primary mt-6 mb-3">
              2.2 Uploaded Media
            </h3>
            <p>
              When you submit media files (images, audio, or video) for
              analysis, we receive and process those files. Uploaded media may
              be temporarily stored on our servers to perform the detection
              analysis. We may retain uploaded media for a limited period to
              support service quality, debugging, and model improvement
              purposes.
            </p>

            <h3 className="text-lg font-heading font-light text-text-primary mt-6 mb-3">
              2.3 Analysis Results
            </h3>
            <p>
              We store the results of your analyses, including detection scores,
              model outputs, and metadata, in your analysis history. This data
              is tied to your account and accessible from your dashboard.
            </p>

            <h3 className="text-lg font-heading font-light text-text-primary mt-6 mb-3">
              2.4 Usage Data &amp; Analytics
            </h3>
            <p>
              We collect analytics and usage data to understand how users
              interact with our platform. This includes:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>Pages visited and features used</li>
              <li>Analysis types and frequency</li>
              <li>API usage patterns and request metadata</li>
              <li>Browser type, operating system, and device information</li>
              <li>IP address and approximate geolocation</li>
              <li>Session duration and interaction patterns</li>
            </ul>

            <h3 className="text-lg font-heading font-light text-text-primary mt-6 mb-3">
              2.5 API Usage Data
            </h3>
            <p>
              If you use the DeepSafe API, we log request metadata including API
              key identifiers, request timestamps, endpoints accessed, response
              codes, and rate limit counters. We do not log the full content of
              API request or response bodies beyond what is necessary for
              billing and abuse prevention.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              3. How We Use Your Information
            </h2>
            <p>We use the information we collect to:</p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>Provide, maintain, and improve the Service</li>
              <li>
                Perform deepfake detection analysis on your submitted media
              </li>
              <li>
                Store your analysis history and provide access to past results
              </li>
              <li>
                Monitor and enforce usage limits, rate limits, and subscription
                quotas
              </li>
              <li>
                Track product analytics to improve user experience and platform
                performance
              </li>
              <li>
                Detect, prevent, and address abuse, fraud, and security
                incidents
              </li>
              <li>
                Communicate with you about your account, service updates, and
                support inquiries
              </li>
              <li>
                Improve our detection accuracy and reliability (see Section 4)
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              4. Media Data &amp; Model Improvement
            </h2>
            <p>
              Uploaded media files may be used to improve the accuracy and
              robustness of our detection models. This may include using
              anonymized or aggregated data for training, validation, and
              benchmarking purposes. We do not publicly share or redistribute
              your uploaded media.
            </p>
            <p className="mt-3">
              If you do not wish for your uploaded media to be used for model
              improvement, please contact us at{" "}
              <a
                href="mailto:contact@deepsafehq.github.io/deepsafe-bench"
                className="text-accent hover:text-accent-hover"
              >
                contact@deepsafehq.github.io/deepsafe-bench
              </a>{" "}
              and we will honor your request.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              5. Data Retention
            </h2>
            <p>We retain your data according to the following guidelines:</p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>
                <span className="text-text-primary">Account information:</span>{" "}
                Retained for the lifetime of your account. Deleted within 30
                days of account deletion.
              </li>
              <li>
                <span className="text-text-primary">Uploaded media:</span> May
                be retained for up to 90 days after analysis for quality
                assurance and service improvement. You may request earlier
                deletion.
              </li>
              <li>
                <span className="text-text-primary">Analysis results:</span>{" "}
                Retained for the lifetime of your account and accessible via
                your dashboard.
              </li>
              <li>
                <span className="text-text-primary">
                  Usage &amp; analytics data:
                </span>{" "}
                Retained in aggregate form indefinitely. Individual-level data
                is retained for up to 12 months.
              </li>
              <li>
                <span className="text-text-primary">API logs:</span> Retained
                for up to 90 days for billing, debugging, and abuse prevention.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              6. Data Security
            </h2>
            <p>
              We implement industry-standard security measures to protect your
              data, including encryption in transit (TLS), access controls,
              secure authentication (JWT, API key hashing), and regular security
              reviews. However, no method of electronic storage or transmission
              is 100% secure, and we cannot guarantee absolute security.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              7. Data Sharing &amp; Third Parties
            </h2>
            <p>
              We do not sell your personal data. We may share your information
              with:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>
                <span className="text-text-primary">Service providers:</span>{" "}
                Third-party vendors that help us operate the Service (e.g.,
                hosting, authentication, analytics). These providers are
                contractually bound to protect your data.
              </li>
              <li>
                <span className="text-text-primary">Legal requirements:</span>{" "}
                When required by law, regulation, legal process, or enforceable
                governmental request.
              </li>
              <li>
                <span className="text-text-primary">
                  Safety &amp; fraud prevention:
                </span>{" "}
                To protect the rights, property, or safety of DeepSafe, our
                users, or the public.
              </li>
              <li>
                <span className="text-text-primary">Business transfers:</span>{" "}
                In connection with a merger, acquisition, or sale of assets,
                your data may be transferred as part of that transaction.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              8. Your Rights
            </h2>
            <p>
              Depending on your jurisdiction, you may have the following rights
              regarding your personal data:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>
                <span className="text-text-primary">Access:</span> Request a
                copy of the personal data we hold about you.
              </li>
              <li>
                <span className="text-text-primary">Correction:</span> Request
                correction of inaccurate or incomplete data.
              </li>
              <li>
                <span className="text-text-primary">Deletion:</span> Request
                deletion of your personal data (&quot;right to be
                forgotten&quot;).
              </li>
              <li>
                <span className="text-text-primary">Data portability:</span>{" "}
                Request your data in a structured, machine-readable format.
              </li>
              <li>
                <span className="text-text-primary">Opt-out:</span> Opt out of
                analytics tracking or marketing communications.
              </li>
              <li>
                <span className="text-text-primary">Restrict processing:</span>{" "}
                Request that we limit the processing of your data under certain
                conditions.
              </li>
            </ul>
            <p className="mt-3">
              To exercise any of these rights, please contact us at{" "}
              <a
                href="mailto:contact@deepsafehq.github.io/deepsafe-bench"
                className="text-accent hover:text-accent-hover"
              >
                contact@deepsafehq.github.io/deepsafe-bench
              </a>
              . We will respond within 30 days.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              9. Cookies &amp; Tracking
            </h2>
            <p>
              We use essential cookies to maintain your session and
              authentication state. We may also use analytics cookies or similar
              technologies to understand how you use the Service. You can
              control cookie preferences through your browser settings.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              10. Children&apos;s Privacy
            </h2>
            <p>
              The Service is not directed at individuals under the age of 16. We
              do not knowingly collect personal data from children. If we become
              aware that we have collected data from a child, we will take steps
              to delete it promptly.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              11. International Data Transfers
            </h2>
            <p>
              Your data may be processed and stored in countries other than your
              own. By using the Service, you consent to the transfer of your
              data to these countries, which may have different data protection
              laws than your jurisdiction.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              12. Changes to This Policy
            </h2>
            <p>
              We may update this Privacy Policy from time to time. We will
              notify you of material changes by posting the updated policy on
              this page and updating the &quot;Last updated&quot; date. Your
              continued use of the Service after changes constitutes acceptance
              of the updated policy.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              13. Contact Us
            </h2>
            <p>
              If you have questions about this Privacy Policy or our data
              practices, please contact us at:
            </p>
            <p className="mt-3 text-text-secondary">
              DeepSafe AI, Inc.
              <br />
              Email:{" "}
              <a
                href="mailto:contact@deepsafehq.github.io/deepsafe-bench"
                className="text-accent hover:text-accent-hover"
              >
                contact@deepsafehq.github.io/deepsafe-bench
              </a>
            </p>
          </section>
        </div>
      </main>

      <PageFooter activePage="privacy" />
    </div>
  );
}
