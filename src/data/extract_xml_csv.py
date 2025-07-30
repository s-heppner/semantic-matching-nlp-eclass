import xml.etree.ElementTree as ET
import pandas as pd
from src.utils.logger import LoggerFactory

# Setup logger
logger = LoggerFactory.get_logger(__name__)

# Extract the classification classes from the raw ECLASS XML data
def extract_eclass_xml(xml_path: str, csv_save_path: str) -> None:
    logger.info(f"Processing XML: {xml_path}")

    try:
        # Load XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML {xml_path}: {e}")
        return

    # Namespaces
    ns = {
        "dic": "urn:eclass:xml-schema:dictionary:5.0",
        "ontoml": "urn:iso:std:iso:is:13584:-32:ed-1:tech:xml-schema:ontoml",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }

    # Extract data from the dictionary
    data = []

    def extract_elements(elements):
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

    dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=ns)
    if dictionary_node is not None:
        for contained in dictionary_node.findall(".//contained_classes", namespaces=ns):
            extract_elements(contained)
    else:
        logger.warning(f"No dictionary node found in {xml_path}")

    # Save results
    df = pd.DataFrame(data)
    df.to_csv(csv_save_path, index=False)
    logger.info(f"Saved CSV to: {csv_save_path}")

# Execute for all segments
exceptions = [] # Adapt manually
for segment in list(range(13, 52)) + [90]:
    if segment in exceptions:
        logger.info(f"Skipping segment {segment}.")
        continue

    xml_path = f"../../data/raw/ECLASS15_0_BASIC_EN_SG_{segment}.xml" # Adapt manually
    save_path = f"../../data/interim/eclass-{segment}.csv" # Adapt manually
    extract_eclass_xml(xml_path, save_path)
