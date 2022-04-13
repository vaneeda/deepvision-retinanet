import uuid
import datetime
from lxml import etree as ET
import csv


def csv2xml(input_xml_path, csv_input_path, orientation):

    def inBounds(point, tl, br):
        x, y = point
        left, top = tl
        right, bottom = br
        if left < x < right and top < y < bottom:
            return True
        return False

    # check if these two boxes have any overlap
    def boxOverlap(bbox1, bbox2):
        # check box2 in box1
        tl = (bbox1['x0'], bbox1['y0'])
        br = (bbox1['x1'], bbox1['y1'])
        for point in [(bbox2['x0'], bbox2['y0']), (bbox2['x1'], bbox2['y1'])]:
            if inBounds(point, tl, br):
                return True
    
        # check box1 in box2
        tl = (bbox2['x0'], bbox2['y0'])
        br = (bbox2['x1'], bbox2['y1'])
        for point in [(bbox1['x0'], bbox1['y0']), (bbox1['x1'], bbox1['y1'])]:
            if inBounds(point, tl, br):
                return True
        # no overlap
        return False

    def getPartialInstances(framedict):
        partials = [False for _ in framedict]
        for act_pos in range(len(partials)-1):
            if not act_pos:
                for next_pos in range(1,len(partials[act_pos:])):
                    np = act_pos + next_pos
                    if boxOverlap(framedict[act_pos], framedict[np]):
                        partials[act_pos] = True
                        partials[np] = True
        return partials

    def indent_xml_elements(elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent_xml_elements(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def create_framework(framedict):
        # Create framework element
        framework_element = ET.Element("framework", attrib={
                "name": Framework_data.name,
                "version": Framework_data.version,
                "species": Framework_data.species,
                "labeltype": Framework_data.labeltype,
                "labeltime": f"{datetime.datetime.now():%Y%m%d%H%M%S%f}"[:-3],  # Get actual timestamp (only 3 ms digits instead of 6)
                "confidence": Framework_data.confidence,
                "annotator": Framework_data.annotator,
                "groundtruth": Framework_data.groundtruth
            })

        # Check
        partial_instances = getPartialInstances(framedict) # Returns boolean list with overlapping objects == true
        # Create object elements
        for fd_id, object_data in enumerate(framedict):
            obj = ET.SubElement(framework_element, "object", attrib={
                "id": str(uuid.uuid1()),
                "real": Object_data.real,
                "species": object_data['species'],
                "length": Object_data.length,
                "labeltime": f"{datetime.datetime.now():%Y%m%d%H%M%S%f}"[:-3], # Get actual timestamp (only 3 ms digits instead of 6)
                "labelmode": Object_data.labelmode,
                "partial": str(partial_instances[fd_id]).lower(),
                "confidence": f"{float(object_data['confidence']):.4f}",
                "annotator": Object_data.annotator,
                "groundtruth": Object_data.groundtruth
            })
            # Create and add bounding box element to object
            ET.SubElement(obj, Object_data.bbox_side,
                attrib={"x0": object_data['x0'], "y0": object_data['y0'], "x1": object_data['x1'], "y1": object_data['y1']})

        return framework_element

    # Initialize static framework information 
    class Framework_data:
        name = "Fish_species_detector_v1.0_Vaneeda_dockerimage"
        version = "1.0"
        species = "None"                        # Default value if no label is given (labeltype != 'image' and  labeltype != 'all')
        labeltype = "object"                    # Fish instances are detected
        confidence = "0"                        # Default value if no confidence is given (labeltype != 'image' and  labeltype != 'all')
        annotator = "fish_species_detector_v1.0_Vaneeda"
        groundtruth = "false"                    # Default value for groundtruth, has to be manually changed after qualitative inspection

    # Initialize static object information
    class Object_data:
        real = "true"                               # All fish instances are real fish (not a man-made object resembling one)
        length = "0"                                # Default length value
        labelmode = "bb"
        partial = "false"                           # Default "partial" flag
        annotator = Framework_data.annotator        # Same value as framework
        groundtruth = Framework_data.groundtruth    # Same value as framework

        if orientation == "Left":
            bbox_side = "lbb"
        else:
            bbox_side = "rbb"
    ######################################
    ### Main script
    ######################################

    # Get input xml file and import element tree
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    # Get csv file and import data to dict
    csvframedict = {}
    with open(csv_input_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['datetime'] in csvframedict:
                csvframedict[row['datetime']].append({"x0": str(round(float(row['x0']))),
                                                      "y0": str(round(float(row['y0']))),
                                                      "x1": str(round(float(row['x1']))),
                                                      "y1": str(round(float(row['y1']))),
                                                      "species": row['label'],
                                                      "confidence": row['score']})
            else:
                csvframedict[row['datetime']] = [{"x0": str(round(float(row['x0']))),
                                                  "y0": str(round(float(row['y0']))),
                                                  "x1": str(round(float(row['x1']))),
                                                  "y1": str(round(float(row['y1']))),
                                                  "species": row['label'],
                                                  "confidence": row['score']}]

    # For each child in csv, update framework
    for child in root[1]:
        if child.attrib["time"] in csvframedict:
            # Generate framework with random data and add it to the element tree
            framework_elem = create_framework(csvframedict[child.attrib["time"]])
            child.insert(-1, framework_elem)

    # print(ET.dump(root)) # print root data

    # Update xml file
    indent_xml_elements(root)
    output_xml_path = input_xml_path.split(".")[0]+"_updated.xml"
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    print(f'CSV data has been exported to {output_xml_path}.')


