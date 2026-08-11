/* Wondering — front end.
 *
 * One request does everything (POST /ingest is synchronous and can take a minute or two),
 * so the waiting state shows a real elapsed clock rather than a fake progress bar.
 */

const $ = (sel) => document.querySelector(sel);

const views = {
  intake: $("#intake"),
  working: $("#working"),
  failure: $("#failure"),
  path: $("#path"),
};

const els = {
  form: $("#form"),
  url: $("#url"),
  submit: $("#submit"),
  reset: $("#reset"),
  retry: $("#retry"),
  clock: $("#clock"),
  workingSource: $("#working-source"),
  status: $("#status"),
  heading: $("#path-heading"),
  sourceLink: $("#path-source-link"),
  progress: $("#progress"),
  stations: $("#stations"),
  template: $("#tpl-station"),
};

/* ─── view switching ─────────────────────────────────────────────────────── */

function show(name) {
  for (const [key, node] of Object.entries(views)) node.hidden = key !== name;
  els.reset.hidden = name === "intake";
}

function announce(message) {
  els.status.textContent = message;
}

/* ─── the clock ──────────────────────────────────────────────────────────── */

let ticking = null;

function startClock() {
  const began = Date.now();
  els.clock.textContent = "0:00";
  stopClock();
  ticking = setInterval(() => {
    const seconds = Math.floor((Date.now() - began) / 1000);
    els.clock.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }, 1000);
}

function stopClock() {
  if (ticking) clearInterval(ticking);
  ticking = null;
}

/* ─── errors ─────────────────────────────────────────────────────────────── */

/* Maps app/api.py's status codes onto something a reader can act on.
   400 = extraction failed, 422 = no captions or a URL the validator refused,
   502 = the model call failed. */
function explain(status, detail) {
  if (status === 422) {
    return {
      head: "Nothing readable in that link",
      body: `${detail} Videos need captions turned on. Articles need to be a page, not a PDF or a paywall.`,
    };
  }
  if (status === 503) {
    return {
      head: "YouTube is blocking this server",
      body: `${detail} YouTube refuses transcript requests from datacenter addresses, which is where this site runs. Article links are unaffected — or run the project locally, where your own connection fetches the transcript.`,
    };
  }
  if (status === 400) {
    return { head: "Couldn't read that page", body: detail };
  }
  if (status === 502) {
    return {
      head: "The model didn't answer",
      body: `${detail} The server needs a working API key for whichever provider it is configured to use.`,
    };
  }
  return { head: "That didn't work", body: detail };
}

function fail(head, body) {
  stopClock();
  $(".failure__head").textContent = head;
  $(".failure__body").textContent = body;
  show("failure");
  announce(`${head}. ${body}`);
  $(".failure__head").focus?.();
}

/* ─── fetching ───────────────────────────────────────────────────────────── */

async function detailOf(response) {
  try {
    const payload = await response.json();
    const detail = payload.detail;
    if (Array.isArray(detail)) return detail[0]?.msg ?? "The URL was rejected.";
    if (typeof detail === "string") return detail;
  } catch {
    /* not JSON — fall through */
  }
  return `The server returned ${response.status}.`;
}

async function build(url) {
  els.workingSource.textContent = url;
  els.submit.disabled = true;
  show("working");
  startClock();
  announce("Building your path. This usually takes 30 to 90 seconds.");

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) {
      const { head, body } = explain(response.status, await detailOf(response));
      fail(head, body);
      return;
    }
    render(await response.json());
  } catch (error) {
    fail("Couldn't reach the server", `${error.message}. Check your connection and try again.`);
  } finally {
    els.submit.disabled = false;
    stopClock();
  }
}

async function openExample(name) {
  show("working");
  els.workingSource.textContent = "Loading a saved path";
  startClock();
  try {
    const response = await fetch(`/examples/${name}`);
    if (!response.ok) {
      fail("That example isn't available", await detailOf(response));
      return;
    }
    render(await response.json());
  } catch (error) {
    fail("Couldn't reach the server", `${error.message}.`);
  } finally {
    stopClock();
  }
}

/* ─── mermaid ────────────────────────────────────────────────────────────── */

/* Loaded lazily and allowed to fail: if the CDN is unreachable or a spec doesn't parse,
   the diagram falls back to its source text, which is already in the markup. */
let mermaidReady = null;

/* Diagram colours are baked into the SVG at render time, so they have to be re-read from
   the stylesheet whenever the palette changes — see the theme listener at the bottom. */
function themeVariables() {
  const styles = getComputedStyle(document.body);
  const token = (name) => styles.getPropertyValue(name).trim();
  return {
    background: token("--ground-2"),
    primaryColor: token("--ground"),
    primaryTextColor: token("--ink"),
    primaryBorderColor: token("--route"),
    lineColor: token("--route"),
    secondaryColor: token("--ground"),
    tertiaryColor: token("--ground"),
    fontSize: "14px",
  };
}

function loadMermaid() {
  mermaidReady ??= import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs")
    .then(({ default: mermaid }) => mermaid);
  return mermaidReady.then((mermaid) => {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      fontFamily: '"IBM Plex Mono", ui-monospace, monospace',
      themeVariables: themeVariables(),
    });
    return mermaid;
  });
}

/* Every diagram currently on the page, so a palette change can redraw them all. */
const drawn = [];

async function drawDiagram(entry) {
  const { canvas, spec, label, index } = entry;
  try {
    const mermaid = await loadMermaid();
    const { svg } = await mermaid.render(`diagram-${index}-${(drawn.epoch = (drawn.epoch ?? 0) + 1)}`, spec);
    canvas.innerHTML = svg;
    const node = canvas.querySelector("svg");
    if (node) {
      node.setAttribute("role", "img");
      node.setAttribute("aria-label", `Diagram: ${label}. The full source is below.`);
      node.removeAttribute("height");
    }
  } catch {
    /* Leave the canvas empty and open the source instead — it says the same thing. */
    canvas.closest(".diagram")?.querySelector("details")?.setAttribute("open", "");
    canvas.remove();
    entry.dead = true;
  }
}

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  for (const entry of drawn) {
    if (!entry.dead && entry.canvas.isConnected) drawDiagram(entry);
  }
});

/* ─── rendering ──────────────────────────────────────────────────────────── */

function paragraphs(text) {
  return text
    .split(/\n{2,}/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
}

function render(path) {
  els.heading.textContent = path.document_title;
  els.sourceLink.textContent = path.source_url;
  els.sourceLink.href = path.source_url;
  els.stations.replaceChildren();

  const lessons = [...path.lessons].sort((a, b) => a.order - b.order);
  const total = lessons.length;
  let answered = 0;
  drawn.length = 0;

  const updateProgress = () => {
    els.progress.textContent = `${answered} of ${total} answered`;
  };
  updateProgress();

  lessons.forEach((lesson, index) => {
    const node = els.template.content.cloneNode(true);
    const station = node.querySelector(".station");
    station.style.setProperty("--i", index);

    node.querySelector(".station__index").textContent =
      `Stop ${String(index + 1).padStart(2, "0")} / ${String(total).padStart(2, "0")}`;
    node.querySelector(".station__title").textContent = lesson.title;
    node.querySelector(".station__cite-text").textContent = lesson.citation;

    node.querySelector(".diagram__src").textContent = lesson.mermaid;

    const prose = node.querySelector(".prose");
    for (const chunk of paragraphs(lesson.explanation)) {
      const p = document.createElement("p");
      p.textContent = chunk;
      prose.append(p);
    }

    wireQuiz(node.querySelector(".quiz"), lesson.exercise, index, () => {
      station.classList.add("is-done");
      answered += 1;
      updateProgress();
    });

    const entry = {
      canvas: node.querySelector(".diagram__canvas"),
      spec: lesson.mermaid,
      label: lesson.title,
      index,
    };
    drawn.push(entry);
    els.stations.append(node);
    drawDiagram(entry);
  });

  show("path");
  announce(`Path built: ${total} lessons from ${path.document_title}.`);
  els.heading.scrollIntoView({ behavior: "smooth", block: "start" });
}

function wireQuiz(form, exercise, index, onAnswered) {
  form.querySelector(".quiz__q").textContent = exercise.question;
  form.querySelector(".quiz__legend").textContent = `Check yourself · stop ${index + 1}`;

  const options = form.querySelector(".quiz__options");
  exercise.options.forEach((text, choice) => {
    const label = document.createElement("label");
    label.className = "opt";

    const input = document.createElement("input");
    input.type = "radio";
    input.name = `q${index}`;
    input.value = String(choice);
    input.required = true;

    const span = document.createElement("span");
    span.textContent = text;

    label.append(input, span);
    options.append(label);
  });

  const verdict = form.querySelector(".quiz__verdict");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (form.classList.contains("is-answered")) return;

    const picked = form.querySelector("input:checked");
    if (!picked) return;

    const chosen = Number(picked.value);
    const correct = exercise.answer_index;
    const right = chosen === correct;

    form.classList.add("is-answered");
    form.querySelectorAll("input").forEach((input) => (input.disabled = true));
    form.querySelector(".btn--check").remove();

    const labels = [...options.children];
    labels[correct]?.classList.add("is-correct");
    if (!right) labels[chosen]?.classList.add("is-wrong");

    verdict.classList.add(right ? "is-right" : "is-wrong");
    const heading = document.createElement("b");
    heading.textContent = right ? "Correct" : `Not quite — the answer is ${exercise.options[correct]}`;
    verdict.replaceChildren(heading, document.createTextNode(exercise.why));

    onAnswered();
  });
}

/* ─── wiring ─────────────────────────────────────────────────────────────── */

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const url = els.url.value.trim();
  if (url) build(url);
});

for (const chip of document.querySelectorAll("[data-example]")) {
  chip.addEventListener("click", () => openExample(chip.dataset.example));
}

function backToIntake() {
  show("intake");
  announce("");
  els.url.focus();
}

els.reset.addEventListener("click", backToIntake);
els.retry.addEventListener("click", backToIntake);
