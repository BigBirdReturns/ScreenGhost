"""Small generic style kernel used before community data is admitted."""
from __future__ import annotations

from experiments.dti.schema import ThemeCard

def seed_theme_cards() -> tuple[ThemeCard, ...]:
    """Small generic style kernel, not a claim of the current official DTI list."""

    rows = [
        ("1920s", ("roaring 20s", "gatsby"), ("1920s", "art deco", "glamour"), ("black", "gold", "jewel"), ("drop waist",), ("fringe", "pearls")),
        ("Y2K", ("2000s", "early 2000s"), ("y2k", "pop", "streetwear"), ("pink", "silver", "denim"), ("fitted", "low rise"), ("metallic", "mini")),
        ("Victorian", ("victorian era",), ("victorian", "historical", "formal"), ("jewel", "dark", "cream"), ("layered", "structured"), ("lace", "high collar")),
        ("Gothic Romance", ("romantic goth", "gothic love"), ("gothic", "romantic", "formal"), ("black", "burgundy", "dark"), ("flowing", "layered"), ("lace", "roses", "choker")),
        ("Dark Academia", ("dark academic",), ("academia", "preppy", "vintage"), ("brown", "black", "earth"), ("layered", "tailored"), ("books", "plaid")),
        ("Cottagecore", ("cottage core",), ("cottagecore", "pastoral", "romantic"), ("pastel", "earth", "cream"), ("flowing", "soft"), ("floral", "ribbon")),
        ("Old Money", ("quiet luxury",), ("old money", "preppy", "luxury"), ("navy", "cream", "neutral"), ("tailored", "clean"), ("pearls", "blazer")),
        ("Kawaii", ("cute",), ("kawaii", "cute", "playful"), ("pastel", "pink", "white"), ("soft", "layered"), ("bows", "hearts")),
        ("Cyberpunk", ("cyber punk",), ("cyberpunk", "futuristic", "edgy"), ("black", "neon", "silver"), ("fitted", "asymmetric"), ("tech", "visor")),
        ("Futuristic Elegance", ("future elegance",), ("futuristic", "formal", "minimalist"), ("silver", "white", "monochrome"), ("sculptural", "clean"), ("metallic", "geometric")),
        ("Regency Era", ("regency", "bridgerton"), ("regency", "historical", "formal"), ("pastel", "cream", "jewel"), ("empire waist", "flowing"), ("gloves", "pearls")),
        ("Renaissance", ("renaissance era",), ("renaissance", "historical", "royal"), ("jewel", "gold", "earth"), ("structured", "voluminous"), ("brocade", "crown")),
        ("Medieval", ("middle ages",), ("medieval", "historical", "fantasy"), ("earth", "jewel", "dark"), ("layered", "flowing"), ("cloak", "belt")),
        ("Greek God or Goddess", ("greek mythology", "greek god", "greek goddess"), ("mythology", "royal", "ethereal"), ("white", "gold", "jewel"), ("draped", "flowing"), ("laurel", "sandals")),
        ("Vampire", ("vampire royalty",), ("vampire", "gothic", "formal"), ("black", "red", "dark"), ("dramatic", "layered"), ("fangs", "cape", "lace")),
        ("Fairy", ("faerie",), ("fairy", "fantasy", "ethereal"), ("pastel", "jewel", "iridescent"), ("flowing", "light"), ("wings", "floral")),
        ("Mermaid", ("under the sea",), ("mermaid", "fantasy", "aquatic"), ("blue", "green", "iridescent"), ("fitted", "flowing"), ("shell", "pearls", "tail")),
        ("Red Carpet", ("award show",), ("red carpet", "formal", "glamour"), ("jewel", "black", "metallic"), ("dramatic", "fitted"), ("jewelry", "train")),
        ("Met Gala", ("gala",), ("avant garde", "formal", "editorial"), ("jewel", "metallic", "monochrome"), ("sculptural", "dramatic"), ("statement", "train")),
        ("Prom", ("prom night",), ("prom", "formal", "youth"), ("pastel", "jewel", "black"), ("fitted", "flowing"), ("corsage", "sparkle")),
        ("Streetwear", ("street wear",), ("streetwear", "casual", "urban"), ("denim", "neutral", "neon"), ("oversized", "layered"), ("sneakers", "cap")),
        ("Monochrome", ("one color",), ("monochrome", "minimalist"), ("monochrome",), ("clean",), ("tonal",)),
        ("Pastel", ("pastel colors",), ("pastel", "soft"), ("pastel",), ("soft", "flowing"), ("bows", "floral")),
        ("Steampunk", ("steam punk",), ("steampunk", "historical", "industrial"), ("brown", "copper", "earth"), ("layered", "structured"), ("gears", "goggles", "corset")),
    ]
    return tuple(
        ThemeCard(
            canonical_name=name,
            aliases=aliases,
            tags=tags,
            palettes=palettes,
            silhouettes=silhouettes,
            motifs=motifs,
            provenance="generic_style_kernel_v1",
        )
        for name, aliases, tags, palettes, silhouettes, motifs in rows
    )
