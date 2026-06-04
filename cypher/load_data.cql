// Quick checks after ingestion.

MATCH (p:Paper)
RETURN count(p) AS papers;

MATCH (s:Section)
RETURN count(s) AS sections;

MATCH (p:Paragraph)
RETURN count(p) AS paragraphs;

MATCH (e:Entity)
RETURN count(e) AS entities;

MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
RETURN a.name, r.relation, b.name
LIMIT 25;

MATCH (p:Paragraph)-[:MENTIONS]->(e:Entity)
RETURN p.paragraph_id, e.name
LIMIT 25;