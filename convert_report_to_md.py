import re
from bs4 import BeautifulSoup, NavigableString, Tag

with open('C:/Users/vyasg/.claude/jobs/755e9ade/tmp/report.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Map base64 images to figure filenames
fig_map = [
    ("fig_training_curves.png", "Figure 7.1: Training and Validation Accuracy/Loss Trajectory across Epochs"),
    ("fig_confusion_matrix.png", "Figure 7.2: Confusion Matrix on 20,000 CIFAKE Test Images (Modified CNN2D)"),
    ("fig_model_benchmarks.png", "Figure 7.3: Accuracy and Parameter Efficiency Benchmarking Across Architectures"),
    ("fig_xai_real_airplane.png", "Figure 7.4: Four-Way Grad-CAM Visual Attribution on Authentic CIFAR-10 Photo"),
    ("fig_xai_fake_airplane.png", "Figure 7.5: Four-Way Grad-CAM Visual Attribution on AI-Generated Synthetic Image"),
]

md_lines = []

# Title & Metadata
md_lines.append("# CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images\n")
md_lines.append("**Academic & Internship Project Report**  ")
md_lines.append("**Student:** Gaurav Vyas  ")
md_lines.append("**Program:** XTRAGRAD Internship  ")
md_lines.append("**Date:** September 2026  ")
md_lines.append("**Reference Paper:** Bird, J.J. & Lotfi, A. (2024). *IEEE Access* / arXiv:2303.14126  \n")
md_lines.append("---\n")

body = soup.find('body')

# Helper to format table
def table_to_md(table_tag):
    rows = table_tag.find_all('tr')
    if not rows:
        return ""
    md_table = []
    headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
    md_table.append("| " + " | ".join(headers) + " |")
    md_table.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows[1:]:
        cols = [td.get_text(strip=True) for td in r.find_all(['td', 'th'])]
        if len(cols) == len(headers):
            md_table.append("| " + " | ".join(cols) + " |")
    return "\n" + "\n".join(md_table) + "\n"

# Iterate top-level elements
img_idx = 0
for elem in body.children:
    if isinstance(elem, NavigableString):
        continue
    if elem.name == 'div' and 'cover' in elem.get('class', []):
        continue # handled above
    if elem.name == 'div' and 'toc' in elem.get('class', []):
        md_lines.append("## Table of Contents\n")
        for li in elem.find_all('li'):
            text = li.get_text(strip=True)
            md_lines.append(f"- {text}")
        md_lines.append("\n---\n")
        continue

    if elem.name == 'h2':
        md_lines.append(f"\n## {elem.get_text(strip=True)}\n")
    elif elem.name == 'h3':
        md_lines.append(f"\n### {elem.get_text(strip=True)}\n")
    elif elem.name == 'p':
        text = elem.get_text(strip=True)
        if text.startswith('Figure '):
            md_lines.append(f"*{text}*\n")
        else:
            md_lines.append(f"{text}\n")
    elif elem.name == 'ul':
        for li in elem.find_all('li'):
            md_lines.append(f"- {li.get_text(strip=True)}")
        md_lines.append("")
    elif elem.name == 'ol':
        for i, li in enumerate(elem.find_all('li'), 1):
            md_lines.append(f"{i}. {li.get_text(strip=True)}")
        md_lines.append("")
    elif elem.name == 'table':
        md_lines.append(table_to_md(elem))
    elif elem.name == 'pre':
        md_lines.append(f"```text\n{elem.get_text()}\n```\n")
    elif elem.name == 'div' and 'eq' in elem.get('class', []):
        md_lines.append(f"\n> **Formula:**  \n> {elem.get_text(strip=True)}\n")
    elif elem.name == 'img':
        if img_idx < len(fig_map):
            fname, cap = fig_map[img_idx]
            md_lines.append(f"\n![{cap}](figures/{fname})\n*{cap}*\n")
            img_idx += 1

out_path = 'D:/project/Internship/XTRAGRAD/CIFAKE/reports/CIFAKE_Project_Report.md'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(md_lines))

print(f"Markdown report generated successfully at: {out_path} ({len(md_lines)} lines)")
