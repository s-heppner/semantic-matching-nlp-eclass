"""Script to extract the classification classes from ECLASS XML files into CSV format."""

import logging
import pandas as pd
import xml.etree.ElementTree as et
from src.utils.logger import LoggerFactory


def extract_eclass_xml(input_path: str, output_path: str, logger: logging.Logger) -> None:
    """Parses an ECLASS XML file and extracts class IDs, names, and definitions into a CSV."""

    # Load ECLASS classes from XML file
    logger.info(f"Processing XML: {input_path}")
    try:
        tree = et.parse(input_path)
        root = tree.getroot()
        logger.info(f"Database loaded from {input_path}.")
    except Exception as e:
        logger.error(f"Failed to parse XML {input_path}: {e}")
        return

    # Namespaces
    ns = {
        "dic": "urn:eclass:xml-schema:dictionary:5.0",
        "ontoml": "urn:iso:std:iso:is:13584:-32:ed-1:tech:xml-schema:ontoml",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    data = []

    def extract_elements(elements):
        """Extracts fields "preferred_name" and "definition" from the given XML elements and appends them to a list."""

        for elem in elements:
            elem_id = elem.attrib.get("id")
            if elem_id is None:
                continue

            # Extract "preferred_name" and "definition" for each classification class
            pref_el = elem.find("preferred_name/label")
            preferred_name = pref_el.text.strip() if pref_el is not None and pref_el.text else None
            def_el = elem.find("definition/text")
            definition = def_el.text.strip() if def_el is not None and def_el.text else None

            # Save results
            data.append({
                "id": elem_id,
                "preferred-name": preferred_name,
                "definition": definition
            })

    # Extract data from the dictionary
    dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=ns)
    if dictionary_node is not None:
        for contained in dictionary_node.findall(".//contained_classes", namespaces=ns):
            extract_elements(contained)
    else:
        logger.warning(f"No dictionary node found in {input_path}")

    # Save results
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    # Settings
    exceptions = []  # Adapt manually: exclude specific segments

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    segments = list(range(13, 52)) + [90]

    # Run for each segment
    for segment in segments:
        if segment in exceptions:
            logger.warning(f"Skipping segment {segment}.")
            continue
        input_path = f"../../data/original/ECLASS15_0_BASIC_EN_SG_{segment}.xml"
        output_path = f"../../data/extracted/eclass-{segment}.csv"
        extract_eclass_xml(input_path, output_path, logger)
