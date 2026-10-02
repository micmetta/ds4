#!/usr/bin/env python3
"""Create a self-contained HTML heatmap viewer for Build 6 transition matrices."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import numpy as np


NUM_EXPERTS = 256


HTML_TEMPLATE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Build 6 - probabilita' di transizione degli expert</title>
  <style>
    body {{ background:#111827; color:#e5e7eb; font-family:system-ui,sans-serif; margin:28px; }}
    h1 {{ margin-bottom:4px; }} p {{ max-width:980px; color:#cbd5e1; }} ul {{ max-width:980px; color:#cbd5e1; }}
    select {{ background:#1f2937; color:#fff; border:1px solid #475569; border-radius:6px; padding:8px; }}
    #layout {{ display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; margin-top:18px; }}
    canvas {{ image-rendering:pixelated; width:min(78vw, 900px); height:min(78vw, 900px); border:1px solid #64748b; }}
    #info {{ min-width:260px; background:#1f2937; border-radius:8px; padding:16px; }}
    code {{ color:#a5f3fc; }} .small {{ font-size:0.9em; color:#94a3b8; }}
  </style>
</head>
<body>
  <h1>Build 6: P<sub>l</sub>(i | j)</h1>
  <p>Ogni cella della heatmap contiene <i>P<sub>l</sub>(i | j)</i>: la probabilita' stimata che l'expert <i>i</i> appaia al token successivo nel layer <i>l</i>, condizionata alla presenza dell'expert <i>j</i> nel token corrente dello stesso layer.</p>
  <p>Ogni heatmap e' una matrice 256 x 256 dove:</p>
  <ul>
    <li>l'asse orizzontale identifica l'expert corrente <i>j</i>, selezionato durante il decoding del token <i>t</i>;</li>
    <li>l'asse verticale identifica l'expert target <i>i</i>, selezionato durante il decoding del token successivo <i>t+1</i>;</li>
    <li><i>l</i> identifica il layer routed considerato.</li>
  </ul>
  <p>Passa il mouse su una cella per leggere il valore esatto.</p>
  <label>Layer routed: <select id="layer"></select></label>
  <div id="layout">
    <canvas id="heatmap" width="256" height="256"></canvas>
    <div id="info"><b>Seleziona una cella</b><p class="small">Il colore mostra la probabilita' condizionata. La scala e' adattata al massimo del layer selezionato per rendere visibili anche le differenze piccole.</p></div>
  </div>
  <p class="small">Artefatto sorgente: <code>{source_name}</code>. Tipo: <code>float32</code>; forma: <code>{shape}</code>.</p>
  <script>
    const encoded = "{encoded_values}";
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    const values = new Float32Array(bytes.buffer);
    const layerFirst = {layer_first};
    const layerCount = {layer_count};
    const experts = {experts};
    const selector = document.getElementById("layer");
    const canvas = document.getElementById("heatmap");
    const context = canvas.getContext("2d");
    const info = document.getElementById("info");

    for (let offset = 0; offset < layerCount; offset++) {{
      const option = document.createElement("option");
      option.value = offset;
      option.textContent = `layer ${{layerFirst + offset}}`;
      selector.appendChild(option);
    }}

    function valueAt(layerOffset, target, current) {{
      return values[(layerOffset * experts + target) * experts + current];
    }}

    function color(value, maximum) {{
      const normalized = maximum > 0 ? Math.min(value / maximum, 1) : 0;
      const red = Math.round(20 + 235 * normalized);
      const green = Math.round(30 + 160 * (1 - normalized));
      const blue = Math.round(80 + 130 * (1 - normalized));
      return [red, green, blue, 255];
    }}

    function draw() {{
      const layerOffset = Number(selector.value);
      let maximum = 0;
      for (let target = 0; target < experts; target++)
        for (let current = 0; current < experts; current++)
          maximum = Math.max(maximum, valueAt(layerOffset, target, current));
      const image = context.createImageData(experts, experts);
      for (let target = 0; target < experts; target++) {{
        for (let current = 0; current < experts; current++) {{
          const pixel = (target * experts + current) * 4;
          image.data.set(color(valueAt(layerOffset, target, current), maximum), pixel);
        }}
      }}
      context.putImageData(image, 0, 0);
      info.innerHTML = `<b>layer ${{layerFirst + layerOffset}}</b><p>Massimo del layer: <code>${{maximum.toExponential(6)}}</code></p><p class="small">Clicca o passa il mouse su una cella.</p>`;
    }}

    function showCell(event) {{
      const rectangle = canvas.getBoundingClientRect();
      const current = Math.min(experts - 1, Math.floor((event.clientX - rectangle.left) * experts / rectangle.width));
      const target = Math.min(experts - 1, Math.floor((event.clientY - rectangle.top) * experts / rectangle.height));
      if (current < 0 || target < 0) return;
      const layer = layerFirst + Number(selector.value);
      const probability = valueAt(Number(selector.value), target, current);
      info.innerHTML = `<b>layer ${{layer}}</b><p><code>P_${{layer}}(${{target}} | ${{current}}) = ${{probability.toExponential(8)}}</code></p><p class="small">Expert corrente j = ${{current}}; expert target i = ${{target}}.</p>`;
    }}

    selector.addEventListener("change", draw);
    canvas.addEventListener("mousemove", showCell);
    canvas.addEventListener("click", showCell);
    draw();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea un report HTML autosufficiente per le matrici P_l(i|j).")
    parser.add_argument("--probabilities", type=Path, required=True, help="Array .npy float32 [layer, target, current].")
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=3, help="Numero del primo layer routed.")
    parser.add_argument("--overwrite", action="store_true", help="Sovrascrive esplicitamente un report HTML esistente.")
    args = parser.parse_args()

    if not args.probabilities.is_file():
        raise FileNotFoundError(f"Matrice non trovata: {args.probabilities}")
    if args.output_html.exists() and not args.overwrite:
        raise FileExistsError(f"Il report esiste gia': {args.output_html}")

    probabilities = np.load(args.probabilities, mmap_mode="r")
    if probabilities.ndim != 3 or probabilities.shape[1:] != (NUM_EXPERTS, NUM_EXPERTS):
        raise ValueError("La matrice deve avere forma [numero_layer, 256, 256].")
    if probabilities.dtype != np.float32:
        raise ValueError("La matrice deve essere salvata in float32.")
    if not np.isfinite(probabilities).all():
        raise ValueError("La matrice contiene valori non finiti.")

    # Base64 makes the report portable: no web server or access to the .npy is needed.
    encoded_values = base64.b64encode(np.ascontiguousarray(probabilities).tobytes()).decode("ascii")
    report = HTML_TEMPLATE.format(
        source_name=args.probabilities.name,
        shape=list(probabilities.shape),
        encoded_values=encoded_values,
        layer_first=args.first_layer,
        layer_count=int(probabilities.shape[0]),
        experts=NUM_EXPERTS,
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(report, encoding="utf-8")
    print("Report HTML delle probabilita' di transizione completato.")
    print(f"Output: {args.output_html}")
    print(f"Dimensione: {args.output_html.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
