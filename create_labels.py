import xml.etree.ElementTree as ET
from utils import *


def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    for image_tag in root.findall(".//image"):
        image_info = {
            'id': image_tag.get('id'),
            'name': image_tag.get('name'),
        }

        for poly_tag in image_tag.findall('.//polygon'):
            poly_info = {
                'label': poly_tag.get('label'),
                'points': poly_tag.get('points'),
            }

            image_info.setdefault('polygons', []).append(poly_info)

        for mask_tag in image_tag.findall('.//mask'):
            mask_info = {
                'label': mask_tag.get('label'),
                'rle': mask_tag.get('rle').split(','),
                'left': int(mask_tag.get('left')),
                'top': int(mask_tag.get('top')),
                'width': int(mask_tag.get('width')),
                'height': int(mask_tag.get('height')),
            }

            image_info.setdefault('masks', []).append(mask_info)

        annotations.append(image_info)

    return annotations


if __name__ == "__main__":
    xml_file_path = os.path.join(info()['general_dir'], 'annotations.xml')
    annotations_data = parse_annotations(xml_file_path)
    # print(annotations_data)

    for image in annotations_data:
        print(f"Image ID: {image['id']}, Image Name: {image['name']}")
        for poly in image.get('polygons', []):
            print(f"  Label: {poly.get('label')}, "
                  f"Coordinates: ({poly.get('points')})")
