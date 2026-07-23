// Render a fenced code block's `title="..."` as a filename caption above it.
//
// mdbook folds the whole code-fence info string into the <code> element's class
// (e.g. class="language-python title=&quot;labs/foo/bar.py&quot;"), but nothing
// in the default theme renders it, so the filename the source declares is
// invisible. This script reads that title and inserts a caption element, so the
// book-wide `title="path"` convention finally shows up for the reader.
(function () {
  function render() {
    var blocks = document.querySelectorAll("pre > code[class*='title=']");
    Array.prototype.forEach.call(blocks, function (code) {
      // Only real code fences carry a filename. Admonish/mermaid blocks are
      // handled by their own preprocessors (and their `title=` is a callout
      // title, not a filename), so never caption them.
      if (/language-(admonish|mermaid)\b/.test(code.className)) return;
      var match = /title="([^"]+)"/.exec(code.className);
      if (!match) return;
      var pre = code.parentElement;
      if (!pre) return;
      var prev = pre.previousElementSibling;
      if (prev && prev.classList.contains("code-filename")) return; // idempotent
      var label = document.createElement("div");
      label.className = "code-filename";
      label.textContent = match[1];
      pre.parentNode.insertBefore(label, pre);
      pre.classList.add("has-filename");
      // Drop the title token from the class so it never leaks into the copy
      // button's text or confuses the syntax highlighter.
      code.className = code.className.replace(/\s*title="[^"]+"/, "");
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
