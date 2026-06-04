CREATE CONSTRAINT paper_id IF NOT EXISTS
FOR (p:Paper)
REQUIRE p.paper_id IS UNIQUE;

CREATE CONSTRAINT section_id IF NOT EXISTS
FOR (s:Section)
REQUIRE s.section_id IS UNIQUE;

CREATE CONSTRAINT paragraph_id IF NOT EXISTS
FOR (p:Paragraph)
REQUIRE p.paragraph_id IS UNIQUE;

CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.name IS UNIQUE;

CREATE INDEX paragraph_text IF NOT EXISTS
FOR (p:Paragraph)
ON (p.text);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity)
ON (e.entity_type);