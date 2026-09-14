"use client";

import Link from "next/link";
import { PageNav } from "@/components/page-nav";
import { PageFooter } from "@/components/page-footer";

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-background text-text-secondary">
      <PageNav />

      {/* Content */}
      <main className="max-w-4xl mx-auto px-6 pt-32 pb-20">
        <h1 className="text-4xl font-heading font-light text-text-primary mb-2">
          Terms of Service
        </h1>
        <p className="text-text-tertiary mb-12">Last updated: March 26, 2026</p>

        <div className="space-y-10 text-[15px] leading-relaxed">
          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              1. Agreement to Terms
            </h2>
            <p>
              By accessing or using the DeepSafe platform at deepsafehq.github.io/deepsafe-bench,
              including the web application, dashboard, REST API, and related
              services (collectively, the &quot;Service&quot;), you agree to be
              bound by these Terms of Service (&quot;Terms&quot;). If you do not
              agree to these Terms, you may not use the Service.
            </p>
            <p className="mt-3">
              These Terms constitute a legally binding agreement between you and
              DeepSafe AI, Inc. (&quot;DeepSafe,&quot; &quot;we,&quot;
              &quot;us,&quot; or &quot;our&quot;).
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              2. Description of Service
            </h2>
            <p>
              DeepSafe is a multi-modal deepfake detection platform that
              analyzes digital media (images, audio, and video) for signs of AI
              generation or manipulation. The Service provides detection scores,
              analysis results, and related insights through a web dashboard and
              a programmatic REST API.
            </p>
            <p className="mt-3">
              DeepSafe uses advanced machine learning to produce its results. No
              detection system is 100% accurate, and results should be
              interpreted as probabilistic assessments, not definitive
              conclusions.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              3. Account Registration
            </h2>
            <p>
              To use certain features of the Service, you must create an
              account. You agree to:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>Provide accurate and complete registration information</li>
              <li>Keep your account credentials secure and confidential</li>
              <li>
                Notify us immediately of any unauthorized access to your account
              </li>
              <li>
                Accept responsibility for all activity that occurs under your
                account
              </li>
            </ul>
            <p className="mt-3">
              We reserve the right to suspend or terminate accounts that violate
              these Terms or that we reasonably believe are being used for
              fraudulent or abusive purposes.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              4. Acceptable Use
            </h2>
            <p>You agree not to use the Service to:</p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>
                Upload content that you do not own or have the right to submit
                for analysis
              </li>
              <li>
                Upload content containing illegal material, including child
                sexual abuse material (CSAM) or content that violates applicable
                laws
              </li>
              <li>
                Attempt to reverse engineer, decompile, or extract the
                underlying models, algorithms, or proprietary technology of the
                Service
              </li>
              <li>
                Circumvent rate limits, authentication mechanisms, or other
                technical restrictions
              </li>
              <li>
                Use the Service to harass, stalk, or intimidate any individual
              </li>
              <li>
                Redistribute, resell, or sublicense access to the Service or its
                results without our written consent
              </li>
              <li>
                Use automated means to access the Service other than through our
                published API
              </li>
              <li>
                Misrepresent DeepSafe analysis results as absolute proof or
                legal evidence without appropriate qualification
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              5. User Content &amp; Ownership
            </h2>
            <p>
              You retain all ownership rights to the media files you upload to
              the Service (&quot;User Content&quot;). By uploading User Content,
              you grant DeepSafe a limited, non-exclusive, worldwide,
              royalty-free license to:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>
                Process, analyze, and store User Content to provide the Service
              </li>
              <li>
                Use anonymized or aggregated data derived from User Content to
                improve our detection models and platform accuracy
              </li>
              <li>
                Generate and store analysis results, metadata, and derived
                insights from User Content
              </li>
            </ul>
            <p className="mt-3">
              We will not publicly share, redistribute, or sell your User
              Content. This license terminates when you delete your User Content
              or your account, except for anonymized data already incorporated
              into aggregate datasets.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              6. API Usage &amp; Rate Limits
            </h2>
            <p>
              Access to the DeepSafe API is subject to your subscription plan.
              Each plan includes a defined number of monthly scans and
              per-minute rate limits. Exceeding these limits may result in
              throttled or rejected requests.
            </p>
            <p className="mt-3">
              API keys are confidential. You are responsible for securing your
              API keys and for all usage associated with them. If you believe an
              API key has been compromised, revoke it immediately via your
              dashboard and contact us.
            </p>
            <p className="mt-3">
              We reserve the right to modify rate limits, quotas, or pricing
              with reasonable notice. Material changes to paid plan terms will
              be communicated at least 30 days in advance.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              7. Subscription &amp; Payments
            </h2>
            <p>
              Certain features of the Service require a paid subscription.
              Subscription fees are billed in advance on a monthly basis.
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>
                Fees are non-refundable except where required by applicable law
              </li>
              <li>
                Unused monthly scan quotas do not roll over to subsequent
                billing periods
              </li>
              <li>We may change pricing with 30 days&apos; advance notice</li>
              <li>
                Failure to pay may result in suspension or downgrade of your
                account
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              8. Intellectual Property
            </h2>
            <p>
              The Service, including its software, models, algorithms,
              documentation, design, and branding, is the exclusive property of
              DeepSafe AI, Inc. and is protected by intellectual property laws.
              Nothing in these Terms grants you any right to use our trademarks,
              logos, or branding without our prior written consent.
            </p>
            <p className="mt-3">
              The analysis results generated by the Service are provided to you
              for your use. However, the underlying detection methodology and
              system architecture remain our proprietary technology.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              9. Disclaimer of Warranties
            </h2>
            <p className="uppercase text-sm text-text-secondary tracking-wide">
              The Service is provided &quot;as is&quot; and &quot;as
              available&quot; without warranties of any kind, whether express or
              implied, including but not limited to implied warranties of
              merchantability, fitness for a particular purpose, and
              non-infringement.
            </p>
            <p className="mt-3">We do not warrant that:</p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>Detection results will be 100% accurate or error-free</li>
              <li>
                The Service will be uninterrupted, secure, or free of defects
              </li>
              <li>
                The Service will meet your specific requirements or expectations
              </li>
              <li>
                Results are suitable for use as legal evidence without
                independent verification
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              10. Limitation of Liability
            </h2>
            <p className="uppercase text-sm text-text-secondary tracking-wide">
              To the maximum extent permitted by law, DeepSafe AI, Inc. shall
              not be liable for any indirect, incidental, special,
              consequential, or punitive damages, including but not limited to
              loss of profits, data, or goodwill, arising out of or in
              connection with your use of the Service.
            </p>
            <p className="mt-3">
              Our total aggregate liability for any claims arising under these
              Terms shall not exceed the amount you paid to us in the twelve
              (12) months preceding the event giving rise to the claim, or one
              hundred dollars ($100), whichever is greater.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              11. Indemnification
            </h2>
            <p>
              You agree to indemnify and hold harmless DeepSafe AI, Inc., its
              officers, directors, employees, and agents from any claims,
              damages, losses, or expenses (including reasonable legal fees)
              arising out of your use of the Service, your violation of these
              Terms, or your infringement of any third-party rights.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              12. Data &amp; Privacy
            </h2>
            <p>
              Your use of the Service is also governed by our{" "}
              <Link
                href="/privacy"
                className="text-accent hover:text-accent-hover"
              >
                Privacy Policy
              </Link>
              , which describes how we collect, use, store, and protect your
              data, including uploaded media and analytics information. By using
              the Service, you acknowledge and agree to the data practices
              described in our Privacy Policy.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              13. Termination
            </h2>
            <p>
              You may terminate your account at any time by contacting us or
              using the account settings in your dashboard. We may suspend or
              terminate your access to the Service at any time, with or without
              cause, including for violation of these Terms.
            </p>
            <p className="mt-3">Upon termination:</p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-text-secondary">
              <li>Your right to access the Service ceases immediately</li>
              <li>
                Your API keys will be revoked and will no longer authenticate
              </li>
              <li>
                We will delete your account data within 30 days, except where
                retention is required by law or for legitimate business purposes
                (e.g., fraud prevention)
              </li>
              <li>
                Anonymized data that has been incorporated into aggregate
                datasets may be retained
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              14. Modifications to Terms
            </h2>
            <p>
              We may modify these Terms at any time. We will notify you of
              material changes by posting the updated Terms on this page and
              updating the &quot;Last updated&quot; date. For material changes
              that affect paid plans, we will provide at least 30 days&apos;
              advance notice.
            </p>
            <p className="mt-3">
              Your continued use of the Service after changes take effect
              constitutes your acceptance of the revised Terms.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              15. Governing Law
            </h2>
            <p>
              These Terms shall be governed by and construed in accordance with
              the laws of the United States, without regard to conflict of law
              principles. Any disputes arising from these Terms or the Service
              shall be resolved through binding arbitration, except where
              prohibited by law.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-heading font-light text-text-primary mb-4">
              16. Contact Us
            </h2>
            <p>
              If you have questions about these Terms, please contact us at:
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

      <PageFooter activePage="terms" />
    </div>
  );
}
