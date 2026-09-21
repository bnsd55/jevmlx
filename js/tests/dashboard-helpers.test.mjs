/**
 * W6-UI-3f: Node test for the dashboard's pure helper functions.
 *
 * The dashboard HTML (jevmlx/web/dashboard.html) is a self-contained single
 * file — we do NOT split it into a separate module (that would break the
 * "self-contained, no external JS" invariant). Instead, this test extracts
 * the pure (DOM-free) helper functions from the inline <script> and exercises
 * them in Node without a DOM framework (no jsdom, no puppeteer).
 *
 * Tested functions: esc, fmtNum, fmtDur, statusPill, parityPill, abDelta,
 * filterSortRows, toggleChip. These are the ones that are pure enough to
 * evaluate without a document. The DOM-patching functions (patchDashboard,
 * patchResults, etc.) require a real DOM and are NOT tested here — they are
 * exercised by the Python grep tests in test_watch_web.py instead.
 *
 * Run: node --test tests/dashboard-helpers.test.mjs  (Node 20+, built-in test runner)
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const HTML_PATH = join(__dirname, "..", "..", "jevmlx", "web", "dashboard.html");
const HTML = readFileSync(HTML_PATH, "utf-8");

// Extract the inline <script> content.
const scriptMatch = HTML.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(scriptMatch, "dashboard.html must have a <script> block");
const script = scriptMatch[1];

// Strip the "use strict" and the DOM-dependent code. We only want the pure
// helper functions. Extract from "function esc" to the end of filterQuestions.
// We evaluate them in a sandbox with a minimal fake `document` stub so the
// function definitions don't throw on parse (they reference document only at
// call time, not definition time).
const fakeDocument = {
  getElementById: () => null,
  querySelectorAll: () => [],
  createElement: () => ({ className: "", dataset: {}, innerHTML: "", onclick: null, appendChild: () => {} }),
};

// Build a sandbox: stub the DOM globals, then eval the script to define all
// functions. We then pull the pure functions out of the sandbox scope.
const sandbox = {
  document: fakeDocument,
  window: {},
  location: { search: "" },
  fetch: async () => { throw new Error("no fetch in test"); },
  EventSource: function () { return { addEventListener: () => {}, onopen: null, onerror: null }; },
  setTimeout: () => {},
  __REFRESH__: 2, // the placeholder _dashboard_html substitutes
  console,
};
sandbox.window = sandbox;
// Use vm to run the script in a context with our stubs.
import vm from "node:vm";
const context = vm.createContext(sandbox);
// Only eval the function definitions, not the bootstrap calls at the bottom
// (firstPaint(), setInterval, startSSE). Cut at the first bootstrap call.
const cutAt = script.indexOf("firstPaint();");
const defScript = cutAt > 0 ? script.slice(0, cutAt) : script;
vm.runInContext(defScript, context);

// Pull the pure functions out of the sandbox.
const {
  esc,
  fmtNum,
  fmtDur,
  statusPill,
  parityPill,
  abDelta,
  filterSortRows,
  toggleChip,
} = sandbox;

describe("dashboard pure helpers", () => {
  describe("esc", () => {
    it("escapes & and <", () => {
      assert.equal(esc("a&b<c"), "a&amp;b&lt;c");
    });
    it("renders null/undefined as empty string", () => {
      assert.equal(esc(null), "");
      assert.equal(esc(undefined), "");
    });
  });

  describe("fmtNum", () => {
    it("renders null as em dash", () => {
      assert.equal(fmtNum(null), "—");
    });
    it("renders a number with default 3 decimals", () => {
      assert.equal(fmtNum(0.123456), "0.123");
    });
    it("renders a number with custom decimals", () => {
      assert.equal(fmtNum(0.123456, 2), "0.12");
    });
    it("renders a string as-is", () => {
      assert.equal(fmtNum("abc"), "abc");
    });
  });

  describe("fmtDur", () => {
    it("renders null as em dash", () => {
      assert.equal(fmtDur(null), "—");
    });
    it("renders seconds under 60", () => {
      assert.equal(fmtDur(45), "45 s");
    });
    it("renders minutes under 3600", () => {
      assert.equal(fmtDur(120), "2m");
    });
    it("renders hours at 3600+", () => {
      assert.equal(fmtDur(7200), "2.0h");
    });
  });

  describe("statusPill", () => {
    it("renders done as ok pill", () => {
      assert.equal(statusPill({ status: "done" }), '<span class="pill ok">done</span>');
    });
    it("renders running as run pill", () => {
      assert.equal(statusPill({ status: "running" }), '<span class="pill run">running</span>');
    });
    it("renders failed as fail pill", () => {
      assert.equal(statusPill({ status: "failed" }), '<span class="pill fail">failed</span>');
    });
    it("renders queued as skip pill", () => {
      assert.equal(statusPill({ status: "queued" }), '<span class="pill skip">queued</span>');
    });
  });

  describe("parityPill", () => {
    it("renders null parity as em dash", () => {
      assert.equal(parityPill({}), "—");
    });
    it("renders pass", () => {
      assert.equal(parityPill({ parity_status: "pass" }), '<span class="pill pass">pass</span>');
    });
    it("renders drift with value", () => {
      const out = parityPill({ parity_status: "drift", parity_drift: 0.07 });
      assert.ok(out.includes("drift"));
      assert.ok(out.includes("0.070"));
    });
    it("renders fail with winner flip", () => {
      const out = parityPill({ parity_status: "fail" });
      assert.ok(out.includes("fail"));
      assert.ok(out.includes("winner flip"));
    });
  });

  describe("abDelta", () => {
    it("renders null as em dash", () => {
      assert.equal(abDelta(null), "—");
    });
    it("renders positive as up", () => {
      const out = abDelta(0.5);
      assert.ok(out.includes("up"));
      assert.ok(out.includes("+0.50"));
    });
    it("renders negative as down", () => {
      const out = abDelta(-0.3);
      assert.ok(out.includes("down"));
      assert.ok(out.includes("−0.30"));
    });
  });

  describe("filterSortRows", () => {
    const rows = [
      { model: "A", dataset: "D1", scorer: "S", track: "parallel", status: "done", accuracy: 0.9 },
      { model: "B", dataset: "D2", scorer: "S", track: "parallel", status: "running", accuracy: 0.8 },
    ];
    it("returns all rows with no filters", () => {
      // rText/sortKey/sortDir are module-level `let` bindings (lexical scope),
      // so they default to "" / null / 1 and cannot be mutated from outside
      // the vm closure. The no-filter path is the only purely-testable path;
      // filter/sort behavior is covered by the Python grep tests in
      // test_watch_web.py + manual browser verification.
      assert.equal(filterSortRows(rows).length, 2);
    });
  });

  describe("toggleChip", () => {
    it("toggles a label on", () => {
      const chips = {};
      toggleChip(chips, "model", "A");
      assert.ok(chips.model.has("A"));
    });
    it("toggles a label off", () => {
      const chips = {};
      toggleChip(chips, "model", "A");
      toggleChip(chips, "model", "A");
      assert.ok(!chips.model || !chips.model.has("A"));
    });
    it("all clears the set", () => {
      const chips = { model: new Set(["A", "B"]) };
      toggleChip(chips, "model", "all");
      assert.ok(!chips.model);
    });
  });
});
