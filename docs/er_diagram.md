# ER / Graph Diagram

```text
Paper
  |
  | HAS_SECTION
  v
Section
  |
  | HAS_PARAGRAPH
  v
Paragraph
  |
  | MENTIONS
  v
Entity

Paragraph
  |
  | supported triples
  v
RELATED_TO edges between Entity nodes