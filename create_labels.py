import xml.etree.ElementTree as ET
from utils import *


def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # get id and name of each of the images
    for image_tag in root.findall(".//image"):
        image_info = {
            'id': image_tag.get('id'),
            'name': image_tag.get('name'),
        }

        # get info for polygons in each image if exists
        for poly_tag in image_tag.findall('.//polygon'):
            poly_info = {
                'label': poly_tag.get('label'),
                'points': poly_tag.get('points'),
            }
            if 'polygons' not in image_info:
                image_info['polygons'] = []
            image_info['polygons'].append(poly_info)

        # get info for masks in each image if exists
        for mask_tag in image_tag.findall('.//mask'):
            mask_info = {
                'label': mask_tag.get('label'),
                'rle': [int(i) for i in mask_tag.get('rle').split(',')],  # convert to list of int from str
                'left': int(mask_tag.get('left')),
                'top': int(mask_tag.get('top')),
                'width': int(mask_tag.get('width')),
                'height': int(mask_tag.get('height')),
            }
            image_info.setdefault('masks', []).append(mask_info)

        annotations.append(image_info)

    return annotations


def create_label_matrix(annotations):
    label_image = []
    for image_info in annotations:
        print(f"Creating labels for Image {image_info['id']} - {image_info['name']}")

        # default width and height of the original images collected are 2000 and 900 respectively
        width = 2000
        height = 900

        # masks are the potato blocks in these annotations
        if 'masks' in image_info:
            print(f"    Number of Mask: {len(image_info.get('masks'))}")
            masked_image = np.array(np.zeros([height, width]), dtype=np.uint8)
            for mask in image_info.get('masks'):
                print(f"{mask['top']}")
                rle, left, top, target_width, target_height = (mask['rle'], mask['left'], mask['top'], mask['width'],
                                                               mask['height'])
                tmp_masked_image = rle2mask(rle, width, height, left, top, target_width, target_height)
                masked_image += tmp_masked_image

            break

        # if masks are not present then there will not be any foliage meaning the image can be called background (0)
        else:
            label_image = np.zeros([height, width], dtype=np.uint8)

    return label_image


if __name__ == "__main__":
    xml_file_path = os.path.join(info()['general_dir'], 'annotations.xml')
    annotations_data = parse_annotations(xml_file_path)
    label = create_label_matrix(annotations_data)
    # print(annotations_data)

    # for image in annotations_data:
    #     print(f"Image ID: {image['id']}, Image Name: {image['name']}")
    #     for poly in image.get('polygons', []):
    #         print(f"  Label: {poly.get('label')}, "
    #               f"Coordinates: ({poly.get('points')})")
