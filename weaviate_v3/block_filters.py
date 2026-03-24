from typing import Optional

from weaviate.classes.query import Filter


CHUNK_BLOCK_TYPES = (
    "Text",
#    "SectionHeader",
    "ListGroup",
    "Table",
    "Code",
    "Caption",
)


def build_block_filter(property_name: str, option: Optional[str] = None):
    normalized = (option or "").strip().lower()
    if not normalized or normalized == "all":
        return None

    if normalized != "chunks":
        raise ValueError(f"Unsupported block filter option: {option}")

    filters = [Filter.by_property(property_name).equal(block_type) for block_type in CHUNK_BLOCK_TYPES]
    return Filter.any_of(filters)
